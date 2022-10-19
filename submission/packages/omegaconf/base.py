import copy
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Type, Union

from antlr4 import ParserRuleContext

from ._utils import (
    _DEFAULT_MARKER_,
    ValueKind,
    _get_value,
    _is_missing_value,
    format_and_raise,
    get_value_kind,
    split_key,
)
from .errors import (
    ConfigKeyError,
    ConfigTypeError,
    InterpolationKeyError,
    InterpolationResolutionError,
    InterpolationToMissingValueError,
    InterpolationValidationError,
    MissingMandatoryValue,
    UnsupportedInterpolationType,
    ValidationError,
)
from .grammar.gen.OmegaConfGrammarParser import OmegaConfGrammarParser
from .grammar_parser import parse
from .grammar_visitor import GrammarVisitor

DictKeyType = Union[str, int, Enum, float, bool]


@dataclass
class Metadata:

    ref_type: Union[Type[Any], Any]

    object_type: Union[Type[Any], Any]

    optional: bool

    key: Any

    # Flags have 3 modes:
    #   unset : inherit from parent (None if no parent specifies)
    #   set to true: flag is true
    #   set to false: flag is false
    flags: Optional[Dict[str, bool]] = None

    # If True, when checking the value of a flag, if the flag is not set None is returned
    # otherwise, the parent node is queried.
    flags_root: bool = False

    resolver_cache: Dict[str, Any] = field(default_factory=lambda: defaultdict(dict))

    def __post_init__(self) -> None:
        if self.flags is None:
            self.flags = {}


@dataclass
class ContainerMetadata(Metadata):
    key_type: Any = None
    element_type: Any = None

    def __post_init__(self) -> None:
        if self.ref_type is None:
            self.ref_type = Any
        assert self.key_type is Any or isinstance(self.key_type, type)
        if self.element_type is not None:
            assert self.element_type is Any or isinstance(self.element_type, type)

        if self.flags is None:
            self.flags = {}


class Node(ABC):
    _metadata: Metadata

    _parent: Optional["Container"]
    _flags_cache: Optional[Dict[str, Optional[bool]]]

    def __init__(self, parent: Optional["Container"], metadata: Metadata):
        self.__dict__["_metadata"] = metadata
        self.__dict__["_parent"] = parent
        self.__dict__["_flags_cache"] = None

    def __getstate__(self) -> Dict[str, Any]:
        # Overridden to ensure that the flags cache is cleared on serialization.
        state_dict = copy.copy(self.__dict__)
        del state_dict["_flags_cache"]
        return state_dict

    def __setstate__(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)
        self.__dict__["_flags_cache"] = None

    def _set_parent(self, parent: Optional["Container"]) -> None:
        assert parent is None or isinstance(parent, Container)
        self.__dict__["_parent"] = parent
        self._invalidate_flags_cache()

    def _invalidate_flags_cache(self) -> None:
        self.__dict__["_flags_cache"] = None

    def _get_parent(self) -> Optional["Container"]:
        parent = self.__dict__["_parent"]
        assert parent is None or isinstance(parent, Container)
        return parent

    def _set_flag(
        self,
        flags: Union[List[str], str],
        values: Union[List[Optional[bool]], Optional[bool]],
    ) -> "Node":
        if isinstance(flags, str):
            flags = [flags]

        if values is None or isinstance(values, bool):
            values = [values]

        if len(values) == 1:
            values = len(flags) * values

        if len(flags) != len(values):
            raise ValueError("Inconsistent lengths of input flag names and values")

        for idx, flag in enumerate(flags):
            value = values[idx]
            if value is None:
                assert self._metadata.flags is not None
                if flag in self._metadata.flags:
                    del self._metadata.flags[flag]
            else:
                assert self._metadata.flags is not None
                self._metadata.flags[flag] = value
        self._invalidate_flags_cache()
        return self

    def _get_node_flag(self, flag: str) -> Optional[bool]:
        """
        :param flag: flag to inspect
        :return: the state of the flag on this node.
        """
        assert self._metadata.flags is not None
        return self._metadata.flags.get(flag)

    def _get_flag(self, flag: str) -> Optional[bool]:
        cache = self.__dict__["_flags_cache"]
        if cache is None:
            cache = self.__dict__["_flags_cache"] = {}

        ret = cache.get(flag, _DEFAULT_MARKER_)
        if ret is _DEFAULT_MARKER_:
            ret = self._get_flag_no_cache(flag)
            cache[flag] = ret
        assert ret is None or isinstance(ret, bool)
        return ret

    def _get_flag_no_cache(self, flag: str) -> Optional[bool]:
        """
        Returns True if this config node flag is set
        A flag is set if node.set_flag(True) was called
        or one if it's parents is flag is set
        :return:
        """
        flags = self._metadata.flags
        assert flags is not None
        if flag in flags and flags[flag] is not None:
            return flags[flag]

        if self._is_flags_root():
            return None

        parent = self._get_parent()
        if parent is None:
            return None
        else:
            # noinspection PyProtectedMember
            return parent._get_flag(flag)

    def _format_and_raise(
        self, key: Any, value: Any, cause: Exception, type_override: Any = None
    ) -> None:
        format_and_raise(
            node=self,
            key=key,
            value=value,
            msg=str(cause),
            cause=cause,
            type_override=type_override,
        )
        assert False

    @abstractmethod
    def _get_full_key(self, key: Optional[Union[DictKeyType, int]]) -> str:
        ...

    def _dereference_node(self) -> "Node":
        node = self._dereference_node_impl(throw_on_resolution_failure=True)
        assert node is not None
        return node

    def _maybe_dereference_node(
        self,
        throw_on_resolution_failure: bool = False,
        memo: Optional[Set[int]] = None,
    ) -> Optional["Node"]:
        return self._dereference_node_impl(
            throw_on_resolution_failure=throw_on_resolution_failure,
            memo=memo,
        )

    def _dereference_node_impl(
        self,
        throw_on_resolution_failure: bool,
        memo: Optional[Set[int]] = None,
    ) -> Optional["Node"]:
        if not self._is_interpolation():
            return self

        parent = self._get_parent()
        if parent is None:
            if throw_on_resolution_failure:
                raise InterpolationResolutionError(
                    "Cannot resolve interpolation for a node without a parent"
                )
            return None
        assert parent is not None
        key = self._key()
        return parent._resolve_interpolation_from_parse_tree(
            parent=parent,
            key=key,
            value=self,
            parse_tree=parse(_get_value(self)),
            throw_on_resolution_failure=throw_on_resolution_failure,
            memo=memo,
        )

    def _get_root(self) -> "Container":
        root: Optional[Container] = self._get_parent()
        if root is None:
            assert isinstance(self, Container)
            return self
        assert root is not None and isinstance(root, Container)
        while root._get_parent() is not None:
            root = root._get_parent()
            assert root is not None and isinstance(root, Container)
        return root

    def _is_missing(self) -> bool:
        """
        Check if the node's value is `???` (does *not* resolve interpolations).
        """
        return _is_missing_value(self)

    def _is_none(self) -> bool:
        """
        Check if the node's value is `None` (does *not* resolve interpolations).
        """
        return self._value() is None

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        ...

    @abstractmethod
    def __ne__(self, other: Any) -> bool:
        ...

    @abstractmethod
    def __hash__(self) -> int:
        ...

    @abstractmethod
    def _value(self) -> Any:
        ...

    @abstractmethod
    def _set_value(self, value: Any, flags: Optional[Dict[str, bool]] = None) -> None:
        ...

    @abstractmethod
    def _is_optional(self) -> bool:
        ...

    @abstractmethod
    def _is_interpolation(self) -> bool:
        ...

    def _key(self) -> Any:
        return self._metadata.key

    def _set_key(self, key: Any) -> None:
        self._metadata.key = key

    def _is_flags_root(self) -> bool:
        return self._metadata.flags_root

    def _set_flags_root(self, flags_root: bool) -> None:
        if self._metadata.flags_root != flags_root:
            self._metadata.flags_root = flags_root
            self._invalidate_flags_cache()


class Container(Node):
    """
    Container tagging interface
    """

    _metadata: ContainerMetadata

    def _get_node(
        self,
        key: Any,
        validate_access: bool = True,
        throw_on_missing_value: bool = False,
        throw_on_missing_key: bool = False,
    ) -> Union[Optional[Node], List[Optional[Node]]]:
        ...

    @abstractmethod
    def __delitem__(self, key: Any) -> None:
        ...

    @abstractmethod
    def __setitem__(self, key: Any, value: Any) -> None:
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        ...

    @abstractmethod
    def __getitem__(self, key_or_index: Any) -> Any:
        ...

    def __copy__(self) -> Any:
        # real shallow copy is impossible because of the reference to the parent.
        return copy.deepcopy(self)

    def _resolve_key_and_root(self, key: str) -> Tuple["Container", str]:
        orig = key
        if not key.startswith("."):
            return self._get_root(), key
        else:
            root: Optional[Container] = self
            assert key.startswith(".")
            while True:
                assert root is not None
                key = key[1:]
                if not key.startswith("."):
                    break
                root = root._get_parent()
                if root is None:
                    raise ConfigKeyError(f"Error resolving key '{orig}'")

            return root, key

    def _select_impl(
        self,
        key: str,
        throw_on_missing: bool,
        throw_on_resolution_failure: bool,
        memo: Optional[Set[int]] = None,
    ) -> Tuple[Optional["Container"], Optional[str], Optional[Node]]:
        """
        Select a value using dot separated key sequence
        """
        from .omegaconf import _select_one

        if key == "":
            return self, "", self

        split = split_key(key)
        root: Optional[Container] = self
        for i in range(len(split) - 1):
            if root is None:
                break

            k = split[i]
            ret, _ = _select_one(
                c=root,
                key=k,
                throw_on_missing=throw_on_missing,
                throw_on_type_error=throw_on_resolution_failure,
            )
            if isinstance(ret, Node):
                ret = ret._maybe_dereference_node(
                    throw_on_resolution_failure=throw_on_resolution_failure,
                    memo=memo,
                )

            if ret is not None and not isinstance(ret, Container):
                parent_key = ".".join(split[0 : i + 1])
                child_key = split[i + 1]
                raise ConfigTypeError(
                    f"Error trying to access {key}: node `{parent_key}` "
                    f"is not a container and thus cannot contain `{child_key}`"
                )
            root = ret

        if root is None:
            return None, None, None

        last_key = split[-1]
        value, _ = _select_one(
            c=root,
            key=last_key,
            throw_on_missing=throw_on_missing,
            throw_on_type_error=throw_on_resolution_failure,
        )
        if value is None:
            return root, last_key, None

        if memo is not None:
            vid = id(value)
            if vid in memo:
                raise InterpolationResolutionError("Recursive interpolation detected")
            # push to memo "stack"
            memo.add(vid)

        try:
            value = root._maybe_resolve_interpolation(
                parent=root,
                key=last_key,
                value=value,
                throw_on_resolution_failure=throw_on_resolution_failure,
                memo=memo,
            )
        finally:
            if memo is not None:
                # pop from memo "stack"
                memo.remove(vid)

        return root, last_key, value

    def _resolve_interpolation_from_parse_tree(
        self,
        parent: Optional["Container"],
        value: "Node",
        key: Any,
        parse_tree: OmegaConfGrammarParser.ConfigValueContext,
        throw_on_resolution_failure: bool,
        memo: Optional[Set[int]],
    ) -> Optional["Node"]:
        """
        Resolve an interpolation.

        This happens in two steps:
            1. The parse tree is visited, which outputs either a `Node` (e.g.,
               for node interpolations "${foo}"), a string (e.g., for string
               interpolations "hello ${name}", or any other arbitrary value
               (e.g., or custom interpolations "${foo:bar}").
            2. This output is potentially validated and converted when the node
               being resolved (`value`) is typed.

        If an error occurs in one of the above steps, an `InterpolationResolutionError`
        (or a subclass of it) is raised, *unless* `throw_on_resolution_failure` is set
        to `False` (in which case the return value is `None`).

        :param parent: Parent of the node being resolved.
        :param value: Node being resolved.
        :param key: The associated key in the parent.
        :param parse_tree: The parse tree as obtained from `grammar_parser.parse()`.
        :param throw_on_resolution_failure: If `False`, then exceptions raised during
            the resolution of the interpolation are silenced, and instead `None` is
            returned.

        :return: A `Node` that contains the interpolation result. This may be an existing
            node in the config (in the case of a node interpolation "${foo}"), or a new
            node that is created to wrap the interpolated value. It is `None` if and only if
            `throw_on_resolution_failure` is `False` and an error occurs during resolution.
        """

        try:
            resolved = self.resolve_parse_tree(
                parse_tree=parse_tree, node=value, key=key, parent=parent, memo=memo
            )
        except InterpolationResolutionError:
            if throw_on_resolution_failure:
                raise
            return None

        return self._validate_and_convert_interpolation_result(
            parent=parent,
            value=value,
            key=key,
            resolved=resolved,
            throw_on_resolution_failure=throw_on_resolution_failure,
        )

    def _validate_and_convert_interpolation_result(
        self,
        parent: Optional["Container"],
        value: "Node",
        key: Any,
        resolved: Any,
        throw_on_resolution_failure: bool,
    ) -> Optional["Node"]:
        from .nodes import AnyNode, InterpolationResultNode, ValueNode

        # If the output is not a Node already (e.g., because it is the output of a
        # custom resolver), then we will need to wrap it within a Node.
        must_wrap = not isinstance(resolved, Node)

        # If the node is typed, validate (and possibly convert) the result.
        if isinstance(value, ValueNode) and not isinstance(value, AnyNode):
            res_value = _get_value(resolved)
            try:
                conv_value = value.validate_and_convert(res_value)
            except ValidationError as e:
                if throw_on_resolution_failure:
                    self._format_and_raise(
                        key=key,
                        value=res_value,
                        cause=e,
                        type_override=InterpolationValidationError,
                    )
                return None

            # If the converted value is of the same type, it means that no conversion
            # was actually needed. As a result, we can keep the original `resolved`
            # (and otherwise, the converted value must be wrapped into a new node).
            if type(conv_value) != type(res_value):
                must_wrap = True
                resolved = conv_value

        if must_wrap:
            return InterpolationResultNode(value=resolved, key=key, parent=parent)
        else:
            assert isinstance(resolved, Node)
            return resolved

    def _validate_not_dereferencing_to_parent(self, node: Node, target: Node) -> None:
        parent: Optional[Node] = node
        while parent is not None:
            if parent is target:
                raise InterpolationResolutionError(
                    "Interpolation to parent node detected"
                )
            parent = parent._get_parent()

    def _resolve_node_interpolation(
        self, inter_key: str, memo: Optional[Set[int]]
    ) -> "Node":
        """A node interpolation is of the form `${foo.bar}`"""
        try:
            root_node, inter_key = self._resolve_key_and_root(inter_key)
        except ConfigKeyError as exc:
            raise InterpolationKeyError(
                f"ConfigKeyError while resolving interpolation: {exc}"
            ).with_traceback(sys.exc_info()[2])

        try:
            parent, last_key, value = root_node._select_impl(
                inter_key,
                throw_on_missing=True,
                throw_on_resolution_failure=True,
                memo=memo,
            )
        except MissingMandatoryValue as exc:
            raise InterpolationToMissingValueError(
                f"MissingMandatoryValue while resolving interpolation: {exc}"
            ).with_traceback(sys.exc_info()[2])

        if parent is None or value is None:
            raise InterpolationKeyError(f"Interpolation key '{inter_key}' not found")
        else:
            self._validate_not_dereferencing_to_parent(node=self, target=value)
            return value

    def _evaluate_custom_resolver(
        self,
        key: Any,
        node: Node,
        inter_type: str,
        inter_args: Tuple[Any, ...],
        inter_args_str: Tuple[str, ...],
    ) -> Any:
        from omegaconf import OmegaConf

        resolver = OmegaConf._get_resolver(inter_type)
        if resolver is not None:
            root_node = self._get_root()
            return resolver(
                root_node,
                self,
                node,
                inter_args,
                inter_args_str,
            )
        else:
            raise UnsupportedInterpolationType(
                f"Unsupported interpolation type {inter_type}"
            )

    def _maybe_resolve_interpolation(
        self,
        parent: Optional["Container"],
        key: Any,
        value: Node,
        throw_on_resolution_failure: bool,
        memo: Optional[Set[int]] = None,
    ) -> Optional[Node]:
        value_kind = get_value_kind(value)
        if value_kind != ValueKind.INTERPOLATION:
            return value

        parse_tree = parse(_get_value(value))
        return self._resolve_interpolation_from_parse_tree(
            parent=parent,
            value=value,
            key=key,
            parse_tree=parse_tree,
            throw_on_resolution_failure=throw_on_resolution_failure,
            memo=memo if memo is not None else set(),
        )

    def resolve_parse_tree(
        self,
        parse_tree: ParserRuleContext,
        node: Node,
        memo: Optional[Set[int]] = None,
        key: Optional[Any] = None,
        parent: Optional["Container"] = None,
    ) -> Any:
        """
        Resolve a given parse tree into its value.

        We make no assumption here on the type of the tree's root, so that the
        return value may be of any type.
        """

        def node_interpolation_callback(
            inter_key: str, memo: Optional[Set[int]]
        ) -> Optional["Node"]:
            return self._resolve_node_interpolation(inter_key=inter_key, memo=memo)

        def resolver_interpolation_callback(
            name: str, args: Tuple[Any, ...], args_str: Tuple[str, ...]
        ) -> Any:
            return self._evaluate_custom_resolver(
                key=key,
                node=node,
                inter_type=name,
                inter_args=args,
                inter_args_str=args_str,
            )

        visitor = GrammarVisitor(
            node_interpolation_callback=node_interpolation_callback,
            resolver_interpolation_callback=resolver_interpolation_callback,
            memo=memo,
        )
        try:
            return visitor.visit(parse_tree)
        except InterpolationResolutionError:
            raise
        except Exception as exc:
            # Other kinds of exceptions are wrapped in an `InterpolationResolutionError`.
            raise InterpolationResolutionError(
                f"{type(exc).__name__} raised while resolving interpolation: {exc}"
            ).with_traceback(sys.exc_info()[2])

    def _re_parent(self) -> None:
        from .dictconfig import DictConfig
        from .listconfig import ListConfig

        # update parents of first level Config nodes to self

        if isinstance(self, Container):
            if isinstance(self, DictConfig):
                content = self.__dict__["_content"]
                if isinstance(content, dict):
                    for _key, value in self.__dict__["_content"].items():
                        if value is not None:
                            value._set_parent(self)
                        if isinstance(value, Container):
                            value._re_parent()
            elif isinstance(self, ListConfig):
                content = self.__dict__["_content"]
                if isinstance(content, list):
                    for item in self.__dict__["_content"]:
                        if item is not None:
                            item._set_parent(self)
                        if isinstance(item, Container):
                            item._re_parent()

    def _invalidate_flags_cache(self) -> None:
        from .dictconfig import DictConfig
        from .listconfig import ListConfig

        # invalidate subtree cache only if the cache is initialized in this node.

        if self.__dict__["_flags_cache"] is not None:
            self.__dict__["_flags_cache"] = None
            if isinstance(self, DictConfig):
                content = self.__dict__["_content"]
                if isinstance(content, dict):
                    for value in self.__dict__["_content"].values():
                        value._invalidate_flags_cache()
            elif isinstance(self, ListConfig):
                content = self.__dict__["_content"]
                if isinstance(content, list):
                    for item in self.__dict__["_content"]:
                        item._invalidate_flags_cache()

    def _has_ref_type(self) -> bool:
        return self._metadata.ref_type is not Any


class SCMode(Enum):
    DICT = 1  # Convert to plain dict
    DICT_CONFIG = 2  # Keep as OmegaConf DictConfig
    INSTANTIATE = 3  # Create a dataclass or attrs class instance
""" Contains a networkx graph representing the underlying powergrid. """
from collections import defaultdict, Counter
from typing import Dict, Tuple, Optional, List, Union, Iterable, Set

import networkx as nx
import numpy as np
from grid2op.Action import ActionSpace
from grid2op.Observation import ObservationSpace, CompleteObservation

LinkInfoType = Tuple[str, int, int, Optional[int], Optional[int], int]


class GridNetworkxGraph:
    """Represents the grid as a networkx graph.

    :param space: Either a grid2op action or observation space object.
    :param fix_links_for_n_sub_steps: Fix one link for substations with less than or equal the number of sub steps + 1.
    :param mask_out_storage_connections: Mask out all storage link actions.
    """
    BUSSES = [1, 2]
    FIXED_BUS = 1

    def __init__(self, space: Union[ObservationSpace, ActionSpace],
                 fix_links_for_n_sub_steps: Optional[int], mask_out_storage_connections: bool):
        self._mask_out_storage_connections = mask_out_storage_connections

        # extract relevant info from observation space
        self.n_sub = space.n_sub
        self.n_line = space.n_line
        self.n_gen = space.n_gen
        self.n_load = space.n_load
        self.n_storage = space.n_storage if hasattr(space, 'n_storage') else 0

        self.load_to_subid = space.load_to_subid
        self.gen_to_subid = space.gen_to_subid
        self.storage_to_subid = space.storage_to_subid if hasattr(space, 'n_storage') else []
        self.or_sub = space.line_or_to_subid
        self.ex_sub = space.line_ex_to_subid
        self.or_topo = space.line_or_pos_topo_vect
        self.ex_topo = space.line_ex_pos_topo_vect
        self.sub_info = space.sub_info

        # Record the number of generators per substation
        self.sub_n_gens = np.zeros_like(self.sub_info)
        for k, v in dict(Counter(list(self.gen_to_subid))).items():
            self.sub_n_gens[k] = v

        # Record the number of loads per substation
        self.sub_n_loads = np.zeros_like(self.sub_info)
        for k, v in dict(Counter(list(self.load_to_subid))).items():
            self.sub_n_loads[k] = v

        self.sub_n_storage = np.zeros_like(self.sub_info)
        for k, v in dict(Counter(list(self.storage_to_subid))).items():
            self.sub_n_storage[k] = v

        self.gen_pos_topo_vect = space.gen_pos_topo_vect
        self.load_pos_topo_vect = space.load_pos_topo_vect
        self.storage_pos_topo_vect = space.storage_pos_topo_vect if hasattr(space, 'n_storage') else []

        # mapping of substation ids to powerlines
        self._sub_ids_to_line_map = defaultdict(list)
        for i_line in range(self.n_line):
            sub_or = space.line_or_to_subid[i_line]
            sub_ex = space.line_ex_to_subid[i_line]
            self._sub_ids_to_line_map[(sub_or, sub_ex)].append(i_line)

        # compile graphs
        self.node_graph = nx.MultiGraph()
        self.line_graph = None
        self._node_idx_to_label: Dict[int, str] = dict()
        self._node_labels_to_idx: Dict[str, int] = dict()
        self._edge_to_link_idx: Dict[Tuple[int, int, int], int] = dict()
        self._link_list = None
        self._link_to_index = defaultdict(list)
        self.substation_id_to_adjacent_links_idx = None
        self.build_graph()

        # Build initial link mask
        self._initial_link_mask = self.build_initial_mask(fix_links_for_n_sub_steps)

        # This can be seeded random.
        self.rng = np.random.RandomState()

    def seed(self, seed: int) -> None:
        """Seed the random number generator used for masking out noop actions."""
        self.rng = np.random.RandomState(seed)

    def build_graph(self) -> None:
        """Compiles the networkx grid2op topology graph."""

        self._add_load_nodes()
        self._add_generator_nodes()
        self._add_storage_nodes()
        self._add_substation_nodes()

        # invert node dictionary
        self._node_labels_to_idx = dict([(v, k) for k, v in self._node_idx_to_label.items()])

        self._add_load_links()
        self._add_generator_links()
        self._add_storage_links()
        self._add_powerline_links()

        assert len(self.node_graph.nodes) == self.n_nodes(), f'{len(self.node_graph.nodes)} vs {self.n_nodes()}'

        # compute edge to link indices
        self._edge_to_link_idx = dict()
        for i, (v, w, j) in enumerate(self.node_graph.edges):
            self._edge_to_link_idx[(v, w, j)] = i

        # convert node to line graph
        self._compile_line_graph()

        self._init_link_list()
        self._init_link_arrays()
        self._init_substation_id_to_adjacent_links_idx()

    def _init_substation_id_to_adjacent_links_idx(self):
        self.sub_station_to_adjacent_links_mask = np.zeros((self.n_sub, len(self._link_list)))
        for sub_station_id in range(self.n_sub):
            node_id_0 = self._sub_node_idx(sub_station_id, 1)
            node_id_1 = self._sub_node_idx(sub_station_id, 2)

            link_connected_to_sub = list()
            for idx, (u, v, k) in enumerate(self.node_graph.edges):
                if node_id_0 in (u, v) or node_id_1 in (u, v):
                    assert idx == self._edge_to_link_idx[(u, v, k)]
                    link_connected_to_sub.append(idx)
            self.sub_station_to_adjacent_links_mask[sub_station_id][np.array(link_connected_to_sub)] = 1

    def _compile_line_graph(self) -> None:
        """Compile the line graph from the node graph"""
        self.line_graph = nx.line_graph(self.node_graph)

    def _add_substation_nodes(self) -> None:
        """Add substation nodes to the graph."""
        # add substations - bus 1
        for sub_idx in range(self.n_sub):
            node_idx = self._sub_node_idx(sub_idx, bus=1)
            label = "S{}-{}".format(sub_idx, 1)
            self._node_idx_to_label[node_idx] = label
            self.node_graph.add_node(node_idx, label=label, node_type="sub", bus=1)

        # add substations - bus 2
        for sub_idx in range(self.n_sub):
            node_idx = self._sub_node_idx(sub_idx, bus=2)
            label = "S{}-{}".format(sub_idx, 2)
            self._node_idx_to_label[node_idx] = label
            self.node_graph.add_node(node_idx, label=label, node_type="sub", bus=2)

    def _sub_node_idx(self, sub_idx: int, bus: int) -> int:
        """Computes a substation node index.
        :param sub_idx: The substation index.
        :param bus: The bus id.
        :return: The index of the node in the link graph.
        """
        return self.n_load + self.n_gen + self.n_storage + sub_idx + (bus - 1) * self.n_sub

    def _add_load_nodes(self) -> None:
        """Adds load nodes to the graph."""
        for load_idx in range(self.n_load):
            node_idx = self._load_node_idx(load_idx)
            label = "L{}".format(load_idx)
            self._node_idx_to_label[node_idx] = label
            self.node_graph.add_node(node_idx, label=label, node_type="load")

    @classmethod
    def _load_node_idx(cls, load_idx: int) -> int:
        """Computes a load node index.
        :param load_idx: The load index.
        :return: The index of the node in the link graph.
        """
        return load_idx

    def _add_generator_nodes(self) -> None:
        """Adds generator nodes to the graph."""
        for gen_idx in range(self.n_gen):
            node_idx = self._generator_node_idx(gen_idx)
            label = "G{}".format(gen_idx)
            self._node_idx_to_label[node_idx] = label
            self.node_graph.add_node(node_idx, label=label, node_type="gen")

    def _generator_node_idx(self, gen_idx: int) -> int:
        """Computes a generator node index.
        :param gen_idx: The generator index.
        :return: The index of the node in the link graph.
        """
        return gen_idx + self.n_load

    def _add_storage_nodes(self) -> None:
        """Adds storage nodes to the graph."""
        for storage_idx in range(self.n_storage):
            node_idx = self._storage_node_idx(storage_idx)
            label = "B{}".format(storage_idx)
            self._node_idx_to_label[node_idx] = label
            self.node_graph.add_node(node_idx, label=label, node_type="storage")

    def _storage_node_idx(self, storage_idx: int) -> int:
        """Computes a storage node index.
        :param storage_idx: The generator index.
        :return: The index of the node in the link graph.
        """
        return self.n_gen + self.n_load + storage_idx

    def _add_load_links(self) -> None:
        """Adds substation load links to the graph."""
        for load_idx in range(self.n_load):
            node_idx = self._load_node_idx(load_idx)
            sub_id = self.load_to_subid[load_idx]
            for bus in [1, 2]:
                target = self._sub_node_idx(sub_id, bus)
                self.node_graph.add_edge(node_idx, target, key=0, link_type="load",
                                         link_info=("load", load_idx, bus, sub_id, None, 0))

    def _add_generator_links(self) -> None:
        """Adds generator load links to the graph."""
        for gen_idx in range(self.n_gen):
            node_idx = self._generator_node_idx(gen_idx)
            sub_id = self.gen_to_subid[gen_idx]
            for bus in [1, 2]:
                target = self._sub_node_idx(sub_id, bus)
                self.node_graph.add_edge(node_idx, target, key=0, link_type="gen",
                                         link_info=("gen", gen_idx, bus, sub_id, None, 0))

    def _add_storage_links(self) -> None:
        """Adds storage load links to the graph."""
        for storage_idx in range(self.n_storage):
            node_idx = self._storage_node_idx(storage_idx)
            sub_id = self.storage_to_subid[storage_idx]
            for bus in [1, 2]:
                target = self._sub_node_idx(sub_id, bus)
                self.node_graph.add_edge(node_idx, target, key=0, link_type="storage",
                                         link_info=("storage", storage_idx, bus, sub_id, None, 0))

    def _add_powerline_links(self) -> None:
        """Adds powerline links to the graph."""

        # Set lines edges
        for line_idx in range(self.n_line):

            # Get substation index for current line
            lor_sub = self.or_sub[line_idx]
            lex_sub = self.ex_sub[line_idx]

            # add bus combinations
            for or_bus, ex_bus in [(1, 1), (2, 2), (1, 2), (2, 1)]:
                # Compute edge vertices indices for current graph
                left_v = self._node_labels_to_idx["S{}-{}".format(lor_sub, or_bus)]
                right_v = self._node_labels_to_idx["S{}-{}".format(lex_sub, ex_bus)]
                edge = (left_v, right_v)

                # Register edge in graph
                link_id = self._sub_ids_to_line_map[(lor_sub, lex_sub)].index(line_idx)
                self.node_graph.add_edge(edge[0], edge[1], key=link_id, link_type="line",
                                         link_info=("line", lor_sub, or_bus, lex_sub, ex_bus, link_id))

    def _init_link_list(self) -> None:
        """Initialized the list of all links in the graph."""
        self._link_list = []
        link_infos = nx.get_edge_attributes(self.node_graph, "link_info")
        for link_idx, edge in enumerate(self.node_graph.edges):
            info = link_infos[edge]
            self._link_list.append(info)

            # write an auxiliary index
            if info[0] == "line":
                self._link_to_index[("line", info[1], info[3], info[5])].append(link_idx)
            elif info[0] == "gen" or info[0] == "load" or info[0] == 'storage':
                self._link_to_index[(info[0], info[1])].append(link_idx)
            else:
                raise ValueError(f'Unidentified edge found in graph: {link_idx} {edge} {info}')
        assert len(self._link_list) == self.n_links(), f'{len(self._link_list)} vs {self.n_links()}'

    def _init_link_arrays(self) -> None:
        """Initialize numpy arrays used to speed up the link_mask() method."""
        # all arrays match the dimensionality of the link space
        n_links = self.n_links()

        # collect the topo vector indices for all links
        self._link_feed_topo_ids = np.zeros(n_links, dtype=np.int32)
        self._link_line_or_topo_ids = np.zeros(n_links, dtype=np.int32)
        self._link_line_ex_topo_ids = np.zeros(n_links, dtype=np.int32)

        # links are statically associated with their origin/extremity bus
        self._link_feed_buses = np.zeros(n_links, dtype=np.int32)
        self._link_line_or_buses = np.zeros(n_links, dtype=np.int32)
        self._link_line_ex_buses = np.zeros(n_links, dtype=np.int32)

        # map links to their affected lines
        self._link_line_ids = np.zeros(n_links, dtype=np.int32)

        # substation ids on the or/ex side for each link
        self._link_sub_id_or = np.zeros(n_links, dtype=np.int32)
        self._link_sub_id_ex = np.zeros(n_links, dtype=np.int32)

        for i_link in range(self.n_links()):

            # extract link and edge data
            link_type, sub_id_or, bus_id_or, sub_id_ex, bus_id_ex, link_id = self._link_list[i_link]
            self._link_sub_id_or[i_link] = sub_id_or
            self._link_sub_id_ex[i_link] = sub_id_ex

            if link_type == "load":
                load_id = sub_id_or
                self._link_feed_topo_ids[i_link] = self.load_pos_topo_vect[load_id]
                self._link_feed_buses[i_link] = bus_id_or
            elif link_type == "gen":
                gen_id = sub_id_or
                self._link_feed_topo_ids[i_link] = self.gen_pos_topo_vect[gen_id]
                self._link_feed_buses[i_link] = bus_id_or
            elif link_type == "storage":
                storage_id = sub_id_or
                self._link_feed_topo_ids[i_link] = self.storage_pos_topo_vect[storage_id]
                self._link_feed_buses[i_link] = bus_id_or
            elif link_type == "line":
                line_idx = self.sub_ids_to_line(sub_id_or, sub_id_ex, link_id)
                self._link_line_ids[i_link] = line_idx

                # set the buses for the respective link line
                self._link_line_or_buses[i_link] = bus_id_or
                self._link_line_ex_buses[i_link] = bus_id_ex

                self._link_line_or_topo_ids[i_link] = self.or_topo[line_idx]
                self._link_line_ex_topo_ids[i_link] = self.ex_topo[line_idx]

    def sub_ids_to_line(self, or_sub_id: int, ex_sub_id: int, link_id: int) -> int:
        """Computes a mapping of two substation ids to a powerline.
        :param or_sub_id: The originating substation.
        :param ex_sub_id: The extremity substation.
        :param link_id: The link id for modeling multiple power lines between two substations
        :return: The powerline id.
        """
        return self._sub_ids_to_line_map[(or_sub_id, ex_sub_id)][link_id]

    def full_link_list(self) -> List[LinkInfoType]:
        """Computes a list of all links in the graph.

        structure: link_type, sub_id_or, bus_id_or, sub_id_ex, bus_id_ex
            - lines: "line", sub_id_or, bus_id_or, sub_id_ex, bus_id_ex
            - loads: "load", load_id, bus_id, None, None
            - gen: "gen", gen_id, bus_id, None, None

        :return: The full link list.
        """
        return self._link_list

    def build_initial_mask(self, fix_links_for_n_sub_steps: Optional[int]) -> np.ndarray:
        """Build the initial link mask to use.

        :param fix_links_for_n_sub_steps: Fix one link for substations with less than or equal the number of sub
                                          steps + 1.

        :return: The initial link mask to use.
        """

        link_mask = np.ones(self.n_links(), dtype=np.float32)
        # Mask out storage connections if applicable
        if self._mask_out_storage_connections:
            for i_link, link in enumerate(self._link_list):
                link_type, sub_id_or, bus_id_or, sub_id_ex, bus_id_ex, link_id = link
                if link_type == "storage":
                    link_mask[i_link] = 0.0

        # Mask out all connections on substation that only have one power line (any change would result in a
        #   blackout)
        for i_link, link in enumerate(self._link_list):
            link_type, sub_id_or, bus_id_or, sub_id_ex, bus_id_ex, link_id = link
            or_line_diff = (self.sub_info[sub_id_or] - self.sub_n_gens[sub_id_or] - self.sub_n_loads[sub_id_or] -
                            self.sub_n_storage[sub_id_or])
            ex_line_diff = (self.sub_info[sub_id_ex] - self.sub_n_gens[sub_id_ex] - self.sub_n_loads[sub_id_ex] -
                            self.sub_n_storage[sub_id_ex])
            if link_type in ['gen', 'load', 'storage'] and ex_line_diff <= 1:
                link_mask[i_link] = 0.0
            elif link_type == 'line' and or_line_diff <= 1 and bus_id_or != self.FIXED_BUS:
                link_mask[i_link] = 0.0
            elif link_type == 'line' and ex_line_diff <= 1 and bus_id_ex != self.FIXED_BUS:
                link_mask[i_link] = 0.0

        if fix_links_for_n_sub_steps is None:
            return link_mask

        fixed_nodes_for_sub = dict()
        for sub_idx, n_sub_connections in enumerate(self.sub_info):
            compare_value = fix_links_for_n_sub_steps + 1
            # If all storage connections are masked out, and the substation is connected to a storage we act as though
            #  the substation is not connected to the storage.
            if self._mask_out_storage_connections and self.sub_n_storage[sub_idx] > 0:
                compare_value += 1
            if n_sub_connections <= compare_value:
                fixed_nodes_for_sub[sub_idx] = None

        for i_link, link in enumerate(self._link_list):
            link_type, sub_id_or, bus_id_or, sub_id_ex, bus_id_ex, link_id = link
            if link_type == "load":
                load_id = sub_id_or
                sub_id = sub_id_ex
                if sub_id in fixed_nodes_for_sub and fixed_nodes_for_sub[sub_id] is None:
                    link_mask[i_link] = 0.0
                    fixed_nodes_for_sub[sub_id] = ('load', load_id)
                elif sub_id in fixed_nodes_for_sub and fixed_nodes_for_sub[sub_id] == ('load', load_id):
                    link_mask[i_link] = 0.0

            elif link_type == "gen":
                gen_id = sub_id_or
                sub_id = sub_id_ex
                if sub_id in fixed_nodes_for_sub and fixed_nodes_for_sub[sub_id] is None:
                    link_mask[i_link] = 0.0
                    fixed_nodes_for_sub[sub_id] = ('gen', gen_id)
                elif sub_id in fixed_nodes_for_sub and fixed_nodes_for_sub[sub_id] == ('gen', gen_id):
                    link_mask[i_link] = 0.0

            elif link_type == "storage" and not self._mask_out_storage_connections:
                storage_id = sub_id_or
                sub_id = sub_id_ex
                if sub_id in fixed_nodes_for_sub and fixed_nodes_for_sub[sub_id] is None:
                    link_mask[i_link] = 0.0
                    fixed_nodes_for_sub[sub_id] = ('storage', storage_id)
                elif sub_id in fixed_nodes_for_sub and fixed_nodes_for_sub[sub_id] == ('storage', storage_id):
                    link_mask[i_link] = 0.0

            elif link_type == "line":
                line_idx = self.sub_ids_to_line(sub_id_or, sub_id_ex, link_id)
                if sub_id_or in fixed_nodes_for_sub and fixed_nodes_for_sub[sub_id_or] is None \
                        and bus_id_or != self.FIXED_BUS:
                    link_mask[i_link] = 0.0
                    fixed_nodes_for_sub[sub_id_or] = ('line', line_idx)
                elif sub_id_or in fixed_nodes_for_sub and fixed_nodes_for_sub[sub_id_or] == ('line', line_idx) \
                        and bus_id_or != self.FIXED_BUS:
                    link_mask[i_link] = 0.0
                elif sub_id_ex in fixed_nodes_for_sub and fixed_nodes_for_sub[sub_id_ex] is None \
                        and bus_id_ex != self.FIXED_BUS:
                    link_mask[i_link] = 0.0
                    fixed_nodes_for_sub[sub_id_ex] = ('line', line_idx)
                elif sub_id_ex in fixed_nodes_for_sub and fixed_nodes_for_sub[sub_id_ex] == ('line', line_idx) \
                        and bus_id_ex != self.FIXED_BUS:
                    link_mask[i_link] = 0.0

        assert all([t is not None for t in fixed_nodes_for_sub.values()]), \
            f'fixed_nodes_for_sub: {[kk for kk, vv in fixed_nodes_for_sub.items() if vv is None]}... ' \
            f'sub_station_config: {[self.sub_info[kk] for kk, vv in fixed_nodes_for_sub.items() if vv is None]}' \
            f'complete: {fixed_nodes_for_sub}'
        return link_mask

    def link_mask(self, state: CompleteObservation) -> np.ndarray:
        """Computes a link mask for the current state.

        :param state: The current state of the environment.
        :return: The link mask.
        """
        link_mask = self._initial_link_mask.copy()
        is_line = self._link_line_or_buses > 0

        # mask out no-ops on feeding points (generators, loads, storage)
        feed_buses = state.topo_vect[self._link_feed_topo_ids]
        # remove link actions matching the current bus id
        link_mask *= 1 - (feed_buses == self._link_feed_buses)

        # mask out line operations based on the current bus id at origin and extremity
        line_or_buses = state.topo_vect[self._link_line_or_topo_ids]
        line_ex_buses = state.topo_vect[self._link_line_ex_topo_ids]
        # identify no-ops of transmission line links
        link_mask *= 1 - (line_or_buses == self._link_line_or_buses) * (line_ex_buses == self._link_line_ex_buses)
        # mask out links that would require changes on two substations
        link_mask *= 1 - (line_or_buses != self._link_line_or_buses) * (
                    line_ex_buses != self._link_line_ex_buses) * is_line

        # mask out inactive power line links
        link_mask *= 1 - (state.line_status[self._link_line_ids] == 0) * is_line
        # mask lines going into maintenance in the next step
        link_mask *= 1 - (state.time_next_maintenance[self._link_line_ids] == 1) * is_line

        # substations need a cooldown before next bus switch
        # cooldown at the extremity is valid for feeding points as well as transmission lines
        link_mask *= 1 - (state.time_before_cooldown_sub[self._link_sub_id_ex] > 0)
        # origin at the origin is only relevant for lines
        link_mask *= 1 - (state.time_before_cooldown_sub[self._link_sub_id_or] > 0) * is_line

        # line cooldown times
        link_mask *= 1 - (state.time_before_cooldown_line[self._link_line_ids] > 0) * is_line

        return link_mask

    def link_node_features_and_mask(self, state: CompleteObservation, with_link_masking: bool) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, nx.Graph]:
        """Computes a link mask, the active node adjacency matrix, a matrix of link features
        and the graph of currently active edges.
        :param state: The current state of the environment.
        :param with_link_masking: If true link masking by connected components checks is computed
        :return: Tuple of (link_mask, active_adjacency, link_features, active_graph)
        """
        link_mask = self._initial_link_mask.copy()
        is_active_link = np.zeros(self.n_links(), dtype=np.bool)
        active_node_adjacency = np.zeros(self._node_adjacency_matrix.shape, dtype=np.float32)
        active_edges = []

        # initialize link feature matrix
        # (link_type[3], line_features[14], active_link[1]])

        # get list of graph edges
        edges = list(self.node_graph.edges)

        # compile full matrix of line features
        line_features = np.transpose([state.p_or, state.q_or, state.v_or, state.a_or,
                                      state.p_ex, state.q_ex, state.v_ex, state.a_ex,
                                      state.rho, state.line_status,
                                      state.timestep_overflow, state.time_before_cooldown_line,
                                      state.time_next_maintenance, state.duration_next_maintenance])

        link_features = np.zeros((self.n_links(), 3 + line_features.shape[1] + 1), dtype=np.float32)

        # mask out invalid line link changes
        noop_idxs = []
        for i_link in np.where(link_mask == 1.0)[0]:

            # extract link and edge data
            link_type, sub_id_or, bus_id_or, sub_id_ex, bus_id_ex, link_id = self._link_list[i_link]
            ei, ej, ek = edges[i_link]

            if link_type == "load":
                load_id = sub_id_or
                bus = state.topo_vect[self.load_pos_topo_vect[load_id]]
                link_status = 0

                if bus_id_or == bus:
                    active_node_adjacency[ei, ej] = 1.0
                    active_node_adjacency[ej, ei] = 1.0
                    active_edges.append((ei, ej, ek))
                    is_active_link[i_link] = True
                    link_status = 1
                    noop_idxs.append(i_link)

                # set link features
                link_features[i_link, 0] = 1  # link type
                link_features[i_link, -1] = link_status  # link active status

            elif link_type == "gen":
                gen_id = sub_id_or
                bus = state.topo_vect[self.gen_pos_topo_vect[gen_id]]
                link_status = 0

                if bus_id_or == bus:
                    active_node_adjacency[ei, ej] = 1.0
                    active_node_adjacency[ej, ei] = 1.0
                    active_edges.append((ei, ej, ek))
                    is_active_link[i_link] = True
                    link_status = 1
                    noop_idxs.append(i_link)

                # set link features
                link_features[i_link, 1] = 1  # link type
                link_features[i_link, -1] = link_status  # link active status

            elif link_type == "storage":
                storage_id = sub_id_or
                bus = state.topo_vect[self.storage_pos_topo_vect[storage_id]]
                link_status = 0

                if bus_id_or == bus:
                    active_node_adjacency[ei, ej] = 1.0
                    active_node_adjacency[ej, ei] = 1.0
                    active_edges.append((ei, ej, ek))
                    is_active_link[i_link] = True
                    link_status = 1
                    noop_idxs.append(i_link)

                # set link features
                link_features[i_link, 1] = 1  # link type
                link_features[i_link, -1] = link_status  # link active status

            elif link_type == "line":
                line_idx = self.sub_ids_to_line(sub_id_or, sub_id_ex, link_id)

                # initialize link status
                link_status = 0

                # -> mask out double link changes

                # get current buses for respective line
                lor_bus = state.topo_vect[self.or_topo[line_idx]]
                lex_bus = state.topo_vect[self.ex_topo[line_idx]]

                if (lor_bus != bus_id_or) and (lex_bus != bus_id_ex):
                    # Mask out links that would require changes on two substations
                    link_mask[i_link] = 0

                elif (lor_bus == bus_id_or) and (lex_bus == bus_id_ex):
                    active_node_adjacency[ei, ej] += 1.0
                    active_node_adjacency[ej, ei] += 1.0
                    active_edges.append((ei, ej, ek))
                    is_active_link[i_link] = True
                    link_status = 1.0
                    link_features[i_link, 3:-1] = line_features[line_idx]  # powerline features
                    noop_idxs.append(i_link)

                # -> mask out inactive power line links
                if bool(state.line_status[line_idx]) is False:
                    link_mask[i_link] = 0

                # set link features
                link_features[i_link, 2] = 1  # link type
                link_features[i_link, -1] = link_status  # link active status

        # mask out noop actions
        if len(noop_idxs):
            link_mask[noop_idxs] = 0

        # check links
        active_graph = None
        if with_link_masking:

            for i_link in np.nonzero(link_mask)[0]:
                # extract link info
                link_type, sub_id_or, bus_id_or, sub_id_ex, bus_id_ex, link_id = self._link_list[i_link]

                if link_type in ["load", "gen", 'storage']:
                    # substations need a cooldown before next bus switch
                    bus, sub_id = bus_id_or, sub_id_ex
                    if sub_id is not None and state.time_before_cooldown_sub[sub_id]:
                        link_mask[i_link] = False
                        continue

                if link_type == "line":

                    # substations need a cooldown before next bus switch
                    if (sub_id_or is not None and state.time_before_cooldown_sub[sub_id_or] or
                            sub_id_ex is not None and state.time_before_cooldown_sub[sub_id_ex]):
                        link_mask[i_link] = False
                        continue

                    # mask lines in cooldown
                    line_id = self.sub_ids_to_line(sub_id_or, sub_id_ex, link_id)
                    if state.time_before_cooldown_line[line_id]:
                        link_mask[i_link] = False
                        continue

                # exclude via clever checks

                # TODO: here we could add additional, more sophisticated checks

                # exclude via simulation

        active_link_adj = self.full_line_adjacency_matrix().copy()
        active_link_adj[np.where(is_active_link == 0.0)] = 0
        active_link_adj[:, np.where(is_active_link == 0.0)] = 0
        return link_mask, active_node_adjacency, active_link_adj, link_features, active_graph

    def full_link_features_and_mask(self, state: CompleteObservation, with_link_masking: bool) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, nx.Graph]:
        """Computes a link mask, the active node adjacency matrix, a matrix of link features
        and the graph of currently active edges.
        :param state: The current state of the environment.
        :param with_link_masking: If true link masking by connected components checks is computed
        :return: Tuple of (link_mask, active_adjacency, link_features, active_graph)
        """
        link_mask = np.ones(self.n_links(), dtype=np.float32)
        is_active_link = np.zeros(self.n_links(), dtype=np.bool)
        active_edges = []

        # get list of graph edges
        edges = list(self.node_graph.edges)

        # compile full matrix of line features
        line_features = np.transpose([state.p_or, state.q_or, state.v_or, state.a_or,
                                      state.p_ex, state.q_ex, state.v_ex, state.a_ex,
                                      state.rho,
                                      state.timestep_overflow, state.time_before_cooldown_line,
                                      state.time_next_maintenance, state.duration_next_maintenance])
        line_features_start = 3
        line_features_end = line_features.shape[1] + line_features_start

        load_features = np.transpose([state.load_p, state.load_q, state.load_v])
        load_features_start = line_features_end
        load_features_end = load_features_start + load_features.shape[1]

        generator_features = np.transpose([state.prod_p, state.prod_q, state.prod_v, state.target_dispatch,
                                           state.actual_dispatch])
        generator_features_start = load_features_end
        generator_features_end = generator_features_start + generator_features.shape[1]

        storage_features = np.transpose([state.storage_charge, state.storage_power, state.storage_power_target,
                                         state.storage_theta])
        storage_features_start = generator_features_end
        storage_features_end = storage_features_start + storage_features.shape[1]

        # initialize link feature matrix
        full_link_features = np.zeros((self.n_links(), 3 + line_features.shape[1] + load_features.shape[1] +
                                       generator_features.shape[1] + storage_features.shape[1] + 1), dtype=np.float32)

        # mask out invalid line link changes
        noop_idxs = []
        for i_link in np.where(link_mask == 1.0)[0]:

            # extract link and edge data
            link_type, sub_id_or, bus_id_or, sub_id_ex, bus_id_ex, link_id = self._link_list[i_link]
            ei, ej, ek = edges[i_link]

            if link_type == "load":
                load_id = sub_id_or
                bus = state.topo_vect[self.load_pos_topo_vect[load_id]]
                link_status = 0

                if bus_id_or == bus:
                    active_edges.append((ei, ej, ek))
                    is_active_link[i_link] = True
                    link_status = 1
                    noop_idxs.append(i_link)
                    full_link_features[i_link, load_features_start:load_features_end] = load_features[load_id]

                # set link features
                full_link_features[i_link, 0] = 1  # link type
                full_link_features[i_link, -1] = link_status  # link active status

            elif link_type == "gen":
                gen_id = sub_id_or
                bus = state.topo_vect[self.gen_pos_topo_vect[gen_id]]
                link_status = 0

                if bus_id_or == bus:
                    active_edges.append((ei, ej, ek))
                    is_active_link[i_link] = True
                    link_status = 1
                    noop_idxs.append(i_link)
                    full_link_features[i_link, generator_features_start: generator_features_end] = generator_features[
                        gen_id]

                # set link features
                full_link_features[i_link, 1] = 1  # link type
                full_link_features[i_link, -1] = link_status  # link active status

            elif link_type == "storage":
                storage_id = sub_id_or
                bus = state.topo_vect[self.storage_pos_topo_vect[storage_id]]
                link_status = 0

                if bus_id_or == bus:
                    active_edges.append((ei, ej, ek))
                    is_active_link[i_link] = True
                    link_status = 1
                    noop_idxs.append(i_link)
                    full_link_features[i_link, storage_features_start: storage_features_end] = storage_features[
                        storage_id]

                # set link features
                full_link_features[i_link, 1] = 1  # link type
                full_link_features[i_link, -1] = link_status  # link active status

            elif link_type == "line":
                line_idx = self.sub_ids_to_line(sub_id_or, sub_id_ex, link_id)

                # initialize link status
                link_status = 0

                # -> mask out double link changes

                # get current buses for respective line
                lor_bus = state.topo_vect[self.or_topo[line_idx]]
                lex_bus = state.topo_vect[self.ex_topo[line_idx]]

                if (lor_bus != bus_id_or) and (lex_bus != bus_id_ex):
                    # Mask out links that would require changes on two substations
                    link_mask[i_link] = 0

                elif (lor_bus == bus_id_or) and (lex_bus == bus_id_ex):
                    active_edges.append((ei, ej, ek))
                    is_active_link[i_link] = True
                    link_status = 1.0
                    full_link_features[i_link, line_features_start:line_features_end] = line_features[
                        line_idx]  # powerline features
                    noop_idxs.append(i_link)

                # -> mask out inactive power line links
                if bool(state.line_status[line_idx]) is False:
                    link_mask[i_link] = 0

                # set link features
                full_link_features[i_link, 2] = 1  # link type
                full_link_features[i_link, -1] = link_status  # link active status

        # mask out noop actions
        if len(noop_idxs):
            link_mask[noop_idxs] = 0

        # check links
        active_graph = None
        if with_link_masking:

            for i_link in np.nonzero(link_mask)[0]:
                # extract link info
                link_type, sub_id_or, bus_id_or, sub_id_ex, bus_id_ex, link_id = self._link_list[i_link]

                if link_type in ["load", "gen", 'storage']:
                    # substations need a cooldown before next bus switch
                    bus, sub_id = bus_id_or, sub_id_ex
                    if sub_id is not None and state.time_before_cooldown_sub[sub_id]:
                        link_mask[i_link] = False
                        continue

                if link_type == "line":

                    # substations need a cooldown before next bus switch
                    if (sub_id_or is not None and state.time_before_cooldown_sub[sub_id_or] or
                            sub_id_ex is not None and state.time_before_cooldown_sub[sub_id_ex]):
                        link_mask[i_link] = False
                        continue

                    # mask lines in cooldown
                    line_id = self.sub_ids_to_line(sub_id_or, sub_id_ex, link_id)
                    if state.time_before_cooldown_line[line_id]:
                        link_mask[i_link] = False
                        continue

        active_link_adj = self.full_line_adjacency_matrix().copy()
        active_link_adj[np.where(is_active_link == 0.0)] = 0
        active_link_adj[:, np.where(is_active_link == 0.0)] = 0
        return link_mask, active_link_adj, full_link_features, active_graph

    def _check_link_connected_component(self,
                                        i_link: int,
                                        active_graph: nx.MultiGraph,
                                        edges: List[Tuple[int, int, int]],
                                        is_active_link: np.ndarray,
                                        bridges: Set[Tuple[int, int]]) -> bool:
        """Check if link change yields a valid grid graph.
        :param i_link: The link index to check.
        :param active_graph: The currently active link graph.
        :param is_active_link: Boolean mask, indicating if the link is currently active
        :param edges: A list of edges.
        :param bridges: A set of cut-edges.
        :return: True if valid; else False
        """

        # extract info of selected link
        link_type, sub_id_or, bus_id_or, sub_id_ex, bus_id_ex, link_id = self._link_list[i_link]
        ei, ej, ek = edges[i_link]
        assert ek == link_id

        if is_active_link[i_link]:
            # noop action
            return True

        # check if edge is not in active graph
        removed_edge = None

        if link_type == "load":
            load_id = sub_id_or
            for j_link in self._link_to_index.get(("load", load_id)):
                if is_active_link[j_link]:
                    removed_edge = edges[j_link]
                    break

        elif link_type == "gen":
            gen_id = sub_id_or
            for j_link in self._link_to_index.get(("gen", gen_id)):
                if is_active_link[j_link]:
                    removed_edge = edges[j_link]
                    break

        elif link_type == "storage":
            gen_id = sub_id_or
            for j_link in self._link_to_index.get(("storage", gen_id)):
                if is_active_link[j_link]:
                    removed_edge = edges[j_link]
                    break

        elif link_type == "line":
            for j_link in self._link_to_index.get(("line", sub_id_or, sub_id_ex, link_id)):
                if is_active_link[j_link]:
                    removed_edge = edges[j_link]
                    break

        if removed_edge is None:
            # no edge will be removed, skip connected component check
            return True

        # if the edge to remove is not a bridge, the connected component check can be skipped
        if tuple(removed_edge[:2]) not in bridges:
            return True

        added_edge = None
        try:
            active_graph.remove_edge(removed_edge[0], removed_edge[1], key=removed_edge[2])
            added_edge = (ei, ej, ek)
            active_graph.add_edge(ei, ej, key=ek)

            # if both nodes only have degree one we have a disconnected component graph
            if active_graph.degree[ei] == 1 and active_graph.degree[ej] == 1:
                return False

            # check if graph is valid and has only one connected component
            # compute connected component expected to contain loads and generators
            con_comp = nx.node_connected_component(active_graph, 0)
            m = self.n_load + self.n_gen

            # check if all other nodes outside of this component have degree 0
            out_off_components_nodes = set(active_graph.nodes).difference(con_comp)
            only_one_graph = True
            for n in out_off_components_nodes:
                if active_graph.degree[n] > 0:
                    only_one_graph = False
                    break

            # check if only substations are outside of connected component
            only_subs_outside_component = False
            if only_one_graph:
                only_subs_outside_component = np.all(np.asarray(list(out_off_components_nodes)) >= m)

            # check if graph is valid
            return only_one_graph and only_subs_outside_component
        finally:
            # revert graph
            if removed_edge is not None:
                active_graph.add_edge(removed_edge[0], removed_edge[1], key=removed_edge[2])
            if added_edge is not None:
                active_graph.remove_edge(added_edge[0], added_edge[1], key=added_edge[2])

    def n_links(self) -> int:
        """Computes the number of links in the graph.
        :return: The link count.
        """
        return self.n_line * 4 + self.n_gen * 2 + self.n_storage * 2 + self.n_load * 2

    def n_nodes(self) -> int:
        """Computes the number of nodes in the graph.
        :return: The node count.
        """
        return self.n_load + self.n_gen + self.n_storage + 2 * self.n_sub


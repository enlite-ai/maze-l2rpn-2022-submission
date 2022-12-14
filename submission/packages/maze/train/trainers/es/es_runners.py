"""Runner implementations for Evolution Strategies"""
import dataclasses
from abc import abstractmethod, ABC
from typing import Union, Optional

from maze.core.agent.policy import Policy
from maze.core.agent.torch_model import TorchModel
from maze.core.utils.factory import Factory
from omegaconf import DictConfig

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.train.trainers.common.model_selection.best_model_selection import BestModelSelection
from maze.train.trainers.common.model_selection.model_selection_base import ModelSelectionBase
from maze.train.trainers.common.training_runner import TrainingRunner
from maze.train.trainers.es.distributed.es_distributed_rollouts import ESDistributedRollouts
from maze.train.trainers.es.distributed.es_dummy_distributed_rollouts import ESDummyDistributedRollouts
from maze.train.trainers.es.distributed.es_subproc_distributed_rollouts import ESSubprocDistributedRollouts
from maze.train.trainers.es.es_shared_noise_table import SharedNoiseTable
from maze.train.trainers.es.es_trainer import ESTrainer
from maze.utils.bcolors import BColors


@dataclasses.dataclass
class ESMasterRunner(TrainingRunner, ABC):
    """Baseclass of ES training master runners (serves as basis for dev and other runners)."""

    shared_noise_table_size: int
    """Number of float values in the deterministically generated pseudo-random table
    (250.000.000 x 32bit floats = 1GB)"""

    shared_noise: Optional[SharedNoiseTable] = dataclasses.field(default=None, init=False)

    @override(TrainingRunner)
    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the training master node.
        """

        super().setup(cfg)

        # --- init the shared noise table ---
        print("********** Init Shared Noise Table **********")
        self.shared_noise = SharedNoiseTable(count=self.shared_noise_table_size)

        # --- initialize policies ---

        torch_policy = TorchPolicy(networks=self._model_composer.policy.networks,
                                   distribution_mapper=self._model_composer.distribution_mapper, device="cpu")
        torch_policy.seed(self.maze_seeding.global_seed)

        # support policy wrapping
        if self._cfg.algorithm.policy_wrapper:
            policy = Factory(Policy).instantiate(
                self._cfg.algorithm.policy_wrapper, torch_policy=torch_policy)
            assert isinstance(policy, Policy) and isinstance(policy, TorchModel)
            torch_policy = policy

        print("********** Trainer Setup **********")
        self._trainer = ESTrainer(
            algorithm_config=cfg.algorithm,
            torch_policy=torch_policy,
            shared_noise=self.shared_noise,
            normalization_stats=self._normalization_statistics
        )

        # initialize model from input_dir
        self._init_trainer_from_input_dir(trainer=self._trainer, state_dict_dump_file=self.state_dict_dump_file,
                                          input_dir=cfg.input_dir)

        self._model_selection = BestModelSelection(dump_file=self.state_dict_dump_file, model=torch_policy,
                                                   dump_interval=self.dump_interval)

    @abstractmethod
    def create_distributed_rollouts(
            self,
            env: Union[StructuredEnv, StructuredEnvSpacesMixin],
            shared_noise: SharedNoiseTable,
            agent_instance_seed: int
    ) -> ESDistributedRollouts:
        """Abstract method, derived runners like ESDevRunner return an appropriate rollout generator.

        :param env: The one and only environment.
        :param shared_noise: Noise table to be shared by all workers.
        :param agent_instance_seed: The agent seed to be used.
        :return: A newly instantiated rollout generator.
        """

    @override(TrainingRunner)
    def run(
            self,
            n_epochs: Optional[int] = None,
            distributed_rollouts: Optional[ESDistributedRollouts] = None,
            model_selection: Optional[ModelSelectionBase] = None
    ) -> None:
        """
        See :py:meth:`~maze.train.trainers.common.training_runner.TrainingRunner.run`.
        :param distributed_rollouts: The distribution interface for experience collection.
        :param n_epochs: Number of epochs to train.
        :param model_selection: Optional model selection class, receives model evaluation results.
        """

        print("********** Run Trainer **********")

        env = self.env_factory()
        env.seed(self.maze_seeding.generate_env_instance_seed())

        # run with pseudo-distribution, without worker processes
        self._trainer.train(
            n_epochs=self._cfg.algorithm.n_epochs if n_epochs is None else n_epochs,
            distributed_rollouts=self.create_distributed_rollouts(
                env=env, shared_noise=self.shared_noise,
                agent_instance_seed=self.maze_seeding.generate_agent_instance_seed()
            ) if distributed_rollouts is None else distributed_rollouts,
            model_selection=self._model_selection if model_selection is None else model_selection
        )


@dataclasses.dataclass
class ESDevRunner(ESMasterRunner):
    """
    Runner config for single-threaded training, based on ESDummyDistributedRollouts.
    """

    n_eval_rollouts: int
    """Fixed number of evaluation runs per epoch."""

    @override(ESMasterRunner)
    def create_distributed_rollouts(
            self, env: Union[StructuredEnv, StructuredEnvSpacesMixin], shared_noise: SharedNoiseTable,
            agent_instance_seed: int,
    ) -> ESDistributedRollouts:
        """use single-threaded rollout generation"""
        return ESDummyDistributedRollouts(env=env, shared_noise=shared_noise, n_eval_rollouts=self.n_eval_rollouts,
                                          agent_instance_seed=agent_instance_seed)


@dataclasses.dataclass
class ESLocalRunner(ESMasterRunner):
    """
    Runner config for multi-process training, based on ESSubprocDistributedRollouts.
    """

    n_train_workers: int
    """Number of worker processes to spawn for training"""

    n_eval_workers: int
    """Number of worker processes to spawn for evaluation"""

    start_method: str
    """Type of start method used for multiprocessing ('forkserver', 'spawn', 'fork')."""

    @override(ESMasterRunner)
    def create_distributed_rollouts(
            self, env: Union[StructuredEnv, StructuredEnvSpacesMixin], shared_noise: SharedNoiseTable,
            agent_instance_seed: int,
    ) -> ESDistributedRollouts:
        """use multi-process rollout generation"""
        BColors.print_colored('Determinism by seeding of the ES algorithm with the Local runner can not be '
                              'guarantied due to the asynchronicity of the implementation.', BColors.WARNING)
        n_workers = self.n_train_workers + self.n_eval_workers
        return ESSubprocDistributedRollouts(
            env_factory=self.env_factory,
            n_training_workers=self.n_train_workers,
            n_eval_workers=self.n_eval_workers,
            shared_noise=self.shared_noise,
            env_seeds=[self.maze_seeding.generate_env_instance_seed() for _ in range(n_workers)],
            agent_seed=agent_instance_seed,
            start_method=self.start_method
        )

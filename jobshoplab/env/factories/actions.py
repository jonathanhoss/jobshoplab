from abc import ABC, abstractmethod
from logging import Logger

from gymnasium import spaces

from jobshoplab.state_machine.time_machines import jump_to_event
from jobshoplab.types import Config, InstanceConfig, State
from jobshoplab.types.action_types import Action, ActionFactoryInfo, ComponentTransition
from jobshoplab.types.state_types import (
    JobState,
    MachineStateState,
    OperationState,
    OperationStateState,
    StateMachineResult,
    TransportState,
    TransportStateState,
)
from jobshoplab.utils import get_logger
from jobshoplab.utils.exceptions import ActionOutOfActionSpace, InvalidValue
from jobshoplab.utils.state_machine_utils import job_type_utils
from jobshoplab.utils.utils import get_id_int


class MinimalActionFactory(ABC):
    def __init__(self) -> None:
        pass

    def interpret(self, action: Action) -> Action:
        return action


class ActionFactory(ABC):
    """
    Abstract base class for action_factorys.
    """

    @abstractmethod
    def __init__(
        self,
        loglevel: int | str,
        config: Config,
        instance: InstanceConfig,
        action_space: spaces.Space,
        *args,
        **kwargs,
    ):
        """
        Initialize the ActionFactory.

        Args:
            loglevel (int): The log level.
            config (Config): The configuration object.
        """
        self.logger: Logger = get_logger(__name__, loglevel)
        self.config: Config = config
        self.instance: InstanceConfig = instance
        self.action_space: spaces.Space = action_space

    def get_dummy_action(self) -> Action:
        return Action(
            transitions=tuple(),
            action_factory_info=ActionFactoryInfo.NoOperation,
            time_machine=jump_to_event,
        )

    @abstractmethod
    def interpret(self, action: spaces.Space, *args, **kwargs) -> Action:
        """
        Interpret the given state.

        Args:
            state (State): The state to interpret.

        Returns:
            StateMachineResult: The result of the interpretation.
        """

    @abstractmethod
    def __repr__(self) -> str:
        """
        Return a string representation of the ActionFactory.

        Returns:
            str: The string representation of the ActionFactory.
        """
        return ""


class DummyActionFactory(ActionFactory):
    """
    A dummy action_factory for testing purposes.
    """

    def __init__(
        self, loglevel: int | str, config: Config, instance: InstanceConfig, *args, **kwargs
    ):
        """
        Initialize the DummyFactory.
        """
        action_space = spaces.Discrete(1)
        super().__init__(loglevel, config, instance, action_space)
        self.logger.info("DummyFactory initialized.")

    def interpret(self, action: spaces.Space) -> Action:
        """
        Interpret the given state.
        """
        self.logger.debug("DummyFactory.interpret() called.")
        return int(action)

    def __repr__(self) -> str:
        """
        Return a string representation of the DummyFactory.
        """
        return f"DummyFactory with action space:{self.action_space}"


class BinaryJobActionFactory(ActionFactory):
    def __init__(
        self,
        loglevel: int | str,
        config: Config,
        instance: InstanceConfig,
        *args,
        **kwargs,
    ):
        self.num_jobs = len(instance.instance.specification)
        action_space = spaces.Discrete(2, start=0)
        super().__init__(loglevel, config, instance, action_space, *args, **kwargs)
        self.dummy_action = Action(
            transitions=tuple(),
            action_factory_info=ActionFactoryInfo.NoOperation,
            time_machine=jump_to_event,
        )

    def interpret(
        self,
        action: int,
        state: StateMachineResult,
        *args,
        **kwargs,
    ) -> Action:
        if len(state.possible_transitions) == 0:
            raise InvalidValue(
                "state.possible_transitions", state.possible_transitions, "No possible transitions"
            )
        transition = state.possible_transitions[0]
        state: State = state.state

        if not self.action_space.contains(action):
            raise ActionOutOfActionSpace(action, self.action_space)
        int_action = int(action)

        if int_action == 0:
            # No Operation Schedule
            self.logger.debug("No Operation Schedule")
            return Action(
                transitions=tuple(),
                action_factory_info=ActionFactoryInfo.NoOperation,
                time_machine=jump_to_event,
            )

        return Action(
            transitions=(transition,),
            action_factory_info=ActionFactoryInfo.Valid,
            time_machine=jump_to_event,
        )

    def __repr__(self) -> str:
        return f"SimpleJsspActionFactory with action space:{self.action_space}"


class MultiDiscreteActionSpaceFactory(ActionFactory):
    def __init__(
        self,
        loglevel: int | str,
        config: Config,
        instance: InstanceConfig,
        *args,
        **kwargs,
    ):
        self.num_jobs = len(instance.instance.specification)
        # action_space = spaces.Discrete(2, start=0)
        action_space = spaces.MultiDiscrete([2] * self.num_jobs)
        super().__init__(loglevel, config, instance, action_space, *args, **kwargs)

    def interpret(
        self,
        action: tuple[int],
        state: State,
        *args,
        **kwargs,
    ) -> Action:
        raise NotImplementedError
        if not self.action_space.contains(action):
            raise ActionOutOfActionSpace(action, self.action_space)

        # Map actions to jobs
        # action 0 is no operation schedule
        # action 1 is job 0, action 2 is job 1, etc.

        transitions = []
        for job_id_int, action in enumerate(action):
            if action == 0:
                continue

            elif action == 1:
                job: JobState = next(
                    filter(lambda j: get_id_int(j.id) == job_id_int, state.jobs), None
                )
                if job is None:
                    raise InvalidValue("job", job_id_int, "Job not found in state.")

                # TODO: Fails if there are no open operations
                next_op = job_type_utils.get_next_not_done_operation(job)

                # getting transporter
                transports = state.transports
                transporter = filter(
                    lambda x: get_id_int(x.id) == get_id_int(job.id),
                    transports,
                )

                transporter: TransportState | None = next(transporter, None)
                if transporter is None:
                    raise InvalidValue(
                        "transporter", next_op.id, "No Transporter found. this is a bug"
                    )
                # make transitions
                component_teleporter = ComponentTransition(
                    component_id=transporter.id,
                    new_state=TransportStateState.WORKING,
                    job_id=job.id,
                )

                component_transition = ComponentTransition(
                    component_id=next_op.machine_id,
                    new_state=MachineStateState.WORKING,
                    job_id=job.id,
                )

                transitions.append(component_teleporter)
                transitions.append(component_transition)

        return Action(
            transitions=tuple(transitions),
            action_factory_info=ActionFactoryInfo.Valid,
        )

    def __repr__(self) -> str:
        return f"SimpleJsspActionFactory with action space:{self.action_space}"


class OperationIndexActionFactory(ActionFactory):
    """
    Action factory that maps a 1D action ID to a specific operation using row-major indexing.

    The action space is a Discrete(n) where n is the total number of operations across all jobs.
    Action IDs map to (job_id, operation_index) pairs using row-major ordering:
    - action 0 -> no-op
    - action 1 -> job 0, operation 0
    - action 2 -> job 0, operation 1
    - ...
    - action (num_ops_job0 + 1) -> job 1, operation 0
    - etc.
    """

    def __init__(
        self,
        loglevel: int | str,
        config: Config,
        instance: InstanceConfig,
        *args,
        **kwargs,
    ):
        # Calculate total operations across all jobs
        self.num_jobs = len(instance.instance.specification)
        self.job_op_counts = [len(job.operations) for job in instance.instance.specification]
        self.total_operations = sum(self.job_op_counts)

        # Action 0 is no-op, actions 1 to total_operations map to specific operations
        action_space = spaces.Discrete(self.total_operations + 1, start=0)
        super().__init__(loglevel, config, instance, action_space, *args, **kwargs)

        # Build lookup table for fast 1D -> 2D conversion
        self._build_action_map()

        self.logger.info(
            "OperationIndexActionFactory initialized with %d operations across %d jobs. Action space: %s",
            self.total_operations,
            self.num_jobs,
            action_space,
        )

    def _build_action_map(self) -> None:
        """Build a mapping from 1D action ID to (job_index, operation_index)."""
        self.action_to_operation = {}
        action_id = 1  # Start from 1 (0 is reserved for no-op)

        for job_idx, num_ops in enumerate(self.job_op_counts):
            for op_idx in range(num_ops):
                self.action_to_operation[action_id] = (job_idx, op_idx)
                action_id += 1

        self.logger.debug("Action map built: %s", self.action_to_operation)

    def _action_to_job_operation(self, action_id: int) -> tuple[int, int] | None:
        """
        Convert 1D action ID to 2D (job_index, operation_index) using row-major indexing.

        Args:
            action_id: The 1D action identifier

        Returns:
            Tuple of (job_index, operation_index) or None if action_id is 0 (no-op)
        """
        if action_id == 0:
            return None
        return self.action_to_operation.get(action_id)

    def interpret(
        self,
        action: int,
        state: StateMachineResult,
        *args,
        **kwargs,
    ) -> Action:
        """
        Interpret the action and return the corresponding Action object.

        Args:
            action: 1D action ID where 0 is no-op and 1+ maps to specific operations
            state: Current state machine result

        Returns:
            Action object with transitions for the selected operation

        Raises:
            ActionOutOfActionSpace: If action is not in the action space
            InvalidValue: If the referenced job or operation is not found or not valid
        """
        if not self.action_space.contains(action):
            raise ActionOutOfActionSpace(action, self.action_space)

        int_action = int(action)
        current_state: State = state.state

        # Handle no-op action
        if int_action == 0:
            self.logger.debug("No Operation (action 0)")
            return Action(
                transitions=tuple(),
                action_factory_info=ActionFactoryInfo.NoOperation,
                time_machine=jump_to_event,
            )

        # Convert 1D action to 2D (job_idx, op_idx)
        coords = self._action_to_job_operation(int_action)
        if coords is None:
            raise InvalidValue(
                "action", int_action, "Action could not be mapped to operation coordinates"
            )

        job_idx, op_idx = coords
        self.logger.debug(f"Action {int_action} -> Job {job_idx}, Operation {op_idx}")

        # Get the job from the state
        if job_idx >= len(current_state.jobs):
            raise InvalidValue(
                "job_idx",
                job_idx,
                f"Job index out of range (total jobs: {len(current_state.jobs)})",
            )

        job: JobState = current_state.jobs[job_idx]

        # Get the operation from the job
        if op_idx >= len(job.operations):
            raise InvalidValue(
                "op_idx",
                op_idx,
                f"Operation index out of range for job {job.id} (total ops: {len(job.operations)})",
            )

        target_operation: OperationState = job.operations[op_idx]

        # Validate that this operation is actionable (IDLE state)
        if target_operation.operation_state_state != OperationStateState.IDLE:
            self.logger.warning(
                f"Operation {target_operation.id} is in state "
                f"{target_operation.operation_state_state}, not IDLE. Returning no-op."
            )
            return Action(
                transitions=tuple(),
                action_factory_info=ActionFactoryInfo.NoMoreOperations,
                time_machine=jump_to_event,
            )

        # Get the transporter for this job
        transporter: TransportState | None = next(
            filter(lambda t: get_id_int(t.id) == get_id_int(job.id), current_state.transports), None
        )

        if transporter is None:
            raise InvalidValue(
                "transporter", job.id, f"No transporter found for job {job.id}. This is a bug."
            )

        # Create transitions for transport and machine
        # transport_transition = ComponentTransition(
        #     component_id=transporter.id,
        #     new_state=TransportStateState.WORKING,
        #     job_id=job.id,
        # )

        machine_transition = ComponentTransition(
            component_id=target_operation.machine_id,
            new_state=MachineStateState.SETUP,
            job_id=job.id,
        )

        self.logger.debug(
            f"Created transitions for job {job.id}, operation {target_operation.id} "
            f"on machine {target_operation.machine_id}"
        )

        return Action(
            transitions=(machine_transition,),
            action_factory_info=ActionFactoryInfo.Valid,
            time_machine=jump_to_event,
        )

    def __repr__(self) -> str:
        return (
            f"OperationIndexActionFactory(jobs={self.num_jobs}, "
            f"total_operations={self.total_operations}, "
            f"action_space={self.action_space})"
        )

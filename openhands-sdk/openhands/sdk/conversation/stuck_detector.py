from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.conversation.types import StuckDetectionThresholds
from openhands.sdk.event import (
    ActionEvent,
    AgentErrorEvent,
    CondensationSummaryEvent,
    Event,
    MessageEvent,
    ObservationBaseEvent,
    ObservationEvent,
)
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


# Maximum recent events to scan for stuck detection.
# This window should be large enough to capture repetitive patterns
# (4 repeats × 2 events per cycle = 8 events minimum, plus buffer for user messages)
MAX_EVENTS_TO_SCAN_FOR_STUCK_DETECTION: int = 20


# One model response can contain several parallel tool calls.  Those calls are one
# agent iteration, not several iterations of a loop.  Keep them together while
# looking for repetition so a batch of sibling failures cannot by itself trip the
# stuck detector.
ResponseBatch = tuple[list[ActionEvent], list[ObservationBaseEvent]]


class StuckDetector:
    """Detects when an agent is stuck in repetitive or unproductive patterns.

    This detector analyzes the conversation history to identify various stuck patterns:
    1. Repeating action-observation cycles
    2. Repeating action-error cycles
    3. Agent monologue (repeated messages without user input)
    4. Repeating alternating action-observation patterns
    5. Context window errors indicating memory issues
    """

    state: ConversationState
    thresholds: StuckDetectionThresholds

    def __init__(
        self,
        state: ConversationState,
        thresholds: StuckDetectionThresholds | None = None,
    ):
        self.state = state
        self.thresholds = thresholds or StuckDetectionThresholds()

    @property
    def action_observation_threshold(self) -> int:
        return self.thresholds.action_observation

    @property
    def action_error_threshold(self) -> int:
        return self.thresholds.action_error

    @property
    def monologue_threshold(self) -> int:
        return self.thresholds.monologue

    @property
    def alternating_pattern_threshold(self) -> int:
        return self.thresholds.alternating_pattern

    def is_stuck(self) -> bool:
        """Check if the agent is currently stuck.

        Note: To avoid materializing potentially large file-backed event histories,
        only the last MAX_EVENTS_TO_SCAN_FOR_STUCK_DETECTION events are analyzed.
        If a user message exists within this window, only events after it are checked.
        Otherwise, all events in the window are analyzed.
        """
        events = list(self.state.events[-MAX_EVENTS_TO_SCAN_FOR_STUCK_DETECTION:])

        # Only look at history after the last user message
        last_user_msg_index = next(
            (
                i
                for i in reversed(range(len(events)))
                if isinstance(events[i], MessageEvent) and events[i].source == "user"
            ),
            -1,  # Default to -1 if no user message found
        )
        if last_user_msg_index != -1:
            events = events[last_user_msg_index + 1 :]

        # Determine minimum events needed
        min_threshold = min(
            self.action_observation_threshold,
            self.action_error_threshold,
            self.monologue_threshold,
        )
        if len(events) < min_threshold:
            return False

        logger.debug(f"Checking for stuck patterns in {len(events)} events")
        logger.debug(
            f"Events after last user message: {[type(e).__name__ for e in events]}"
        )

        response_batches = self._get_complete_response_batches(events)

        # Check all stuck patterns
        # scenario 1: same action, same observation
        if self._is_stuck_repeating_action_observation(response_batches):
            return True

        # scenario 2: same action, errors
        if self._is_stuck_repeating_action_error(response_batches):
            return True

        # scenario 3: monologue
        if self._is_stuck_monologue(events):
            return True

        # scenario 4: action, observation alternating pattern
        if len(events) >= self.alternating_pattern_threshold:
            if self._is_stuck_alternating_action_observation(response_batches):
                return True

        # scenario 5: context window error loop
        if len(events) >= 10:
            if self._is_stuck_context_window_error(events):
                return True

        return False

    def _get_complete_response_batches(
        self, events: list[Event]
    ) -> list[ResponseBatch]:
        """Group actions and their observations by originating LLM response.

        OpenHands emits one :class:`ActionEvent` per tool call, even when several
        calls came from the same model response.  Treating those siblings as
        separate loop iterations causes false-positive stuck detection (especially
        when a parallel batch contains several invalid calls).  A batch is included
        only when every action visible in the scan window has a corresponding
        observation; partial batches at the edge of the bounded history are ignored.
        """
        actions_by_response: dict[str, list[ActionEvent]] = {}
        observations_by_tool_call: dict[str, list[ObservationBaseEvent]] = {}

        for event in events:
            if not isinstance(event, ActionEvent):
                continue
            response_id = str(event.llm_response_id)
            actions_by_response.setdefault(response_id, []).append(event)

        for event in events:
            if not isinstance(event, ObservationBaseEvent):
                continue
            observations_by_tool_call.setdefault(str(event.tool_call_id), []).append(
                event
            )

        batches: list[ResponseBatch] = []
        for actions in actions_by_response.values():
            if not all(
                str(action.tool_call_id) in observations_by_tool_call
                for action in actions
            ):
                continue

            # Parallel executors may complete siblings in any order.  Canonicalize
            # their observations back into model action order before comparing two
            # response batches.
            observations = [
                observation
                for action in actions
                for observation in observations_by_tool_call[str(action.tool_call_id)]
            ]
            batches.append((actions, observations))
        return batches

    def _is_stuck_repeating_action_observation(
        self, response_batches: list[ResponseBatch]
    ) -> bool:
        # scenario 1: same action, same observation
        threshold = self.action_observation_threshold

        # Check for a loop of identical response-level action-observation batches.
        last_batches = response_batches[-threshold:]
        if len(last_batches) >= threshold:
            logger.debug(
                f"Found {len(response_batches)} complete response batches, "
                "checking for equality"
            )
            actions_equal = all(
                self._event_lists_eq(last_batches[0][0], batch[0])
                for batch in last_batches[1:]
            )
            observations_equal = all(
                self._event_lists_eq(last_batches[0][1], batch[1])
                for batch in last_batches[1:]
            )
            logger.debug(
                f"Actions equal: {actions_equal}, "
                f"Observations equal: {observations_equal}"
            )

            if actions_equal and observations_equal:
                logger.warning("Action, Observation loop detected")
                return True
        else:
            logger.debug(
                f"Not enough complete response batches: {len(response_batches)} batches"
            )

        return False

    def _is_stuck_repeating_action_error(
        self, response_batches: list[ResponseBatch]
    ) -> bool:
        # scenario 2: same action, errors
        threshold = self.action_error_threshold
        if len(response_batches) < threshold:
            return False

        last_batches = response_batches[-threshold:]

        # Are the last N response-level action batches the "same"?
        if all(
            self._event_lists_eq(last_batches[0][0], batch[0])
            for batch in last_batches[1:]
        ):
            # And did every action in every response batch produce an error?
            if all(
                isinstance(obs, AgentErrorEvent)
                for _actions, observations in last_batches
                for obs in observations
            ):
                logger.warning("Action, Error loop detected")
                return True

        # Check if observations are errors
        return False

    def _is_stuck_monologue(self, events: list[Event]) -> bool:
        # scenario 3: monologue
        # check for repeated MessageActions with source=AGENT
        # see if the agent is engaged in a good old monologue, telling
        # itself the same thing over and over
        threshold = self.monologue_threshold
        if len(events) < threshold:
            return False

        # Look for N consecutive agent messages without user interruption
        agent_message_count = 0

        for event in reversed(events):
            if isinstance(event, MessageEvent):
                if event.source == "agent":
                    agent_message_count += 1
                elif event.source == "user":
                    break  # User interrupted, not a monologue
            elif isinstance(event, CondensationSummaryEvent):
                # Condensation events don't break the monologue pattern
                continue
            else:
                # Other events (actions/observations) don't count as monologue
                break

        return agent_message_count >= threshold

    def _is_stuck_alternating_action_observation(
        self, response_batches: list[ResponseBatch]
    ) -> bool:
        # scenario 4: alternating action-observation loop
        threshold = self.alternating_pattern_threshold

        last_batches = response_batches[-threshold:]
        if len(last_batches) == threshold:
            # Check alternating pattern: [A, B, A, B, A, B] where even/odd match
            actions_equal = all(
                self._event_lists_eq(last_batches[i][0], last_batches[i + 2][0])
                for i in range(threshold - 2)
            )
            observations_equal = all(
                self._event_lists_eq(last_batches[i][1], last_batches[i + 2][1])
                for i in range(threshold - 2)
            )

            if actions_equal and observations_equal:
                logger.warning("Alternating Action, Observation loop detected")
                return True

        return False

    def _event_lists_eq(self, events1: list[Event], events2: list[Event]) -> bool:
        return len(events1) == len(events2) and all(
            self._event_eq(event1, event2)
            for event1, event2 in zip(events1, events2, strict=True)
        )

    def _is_stuck_context_window_error(self, _events: list[Event]) -> bool:
        """Detects if we are stuck in a loop of context window errors.

        This happens when we repeatedly get context window errors and try to trim,
        but the trimming does not work, causing us to get more context window errors.
        The pattern is repeated AgentCondensationObservation events without any other
        events between them.
        """
        # TODO: blocked by https://github.com/OpenHands/agent-sdk/issues/282
        return False

    def _event_eq(self, event1: Event, event2: Event) -> bool:
        """
        Compare two events for equality, ignoring irrelevant
        details like ids, metrics.
        """
        # Must be same type
        if type(event1) is not type(event2):
            return False

        # For ActionEvents, compare the action content, ignoring IDs
        if isinstance(event1, ActionEvent) and isinstance(event2, ActionEvent):
            base_equal = (
                event1.source == event2.source
                and event1.thought == event2.thought
                and event1.action == event2.action
                and event1.tool_name == event2.tool_name
                # Ignore tool_call_id, llm_response_id, action_id as they vary
            )
            if not base_equal:
                return False
            # Invalid or unknown tools have ``action=None``.  Their raw tool-call
            # arguments are still semantically relevant; ignoring them makes
            # parallel calls to the same tool with different arguments look equal.
            if event1.action is None and event2.action is None:
                return event1.tool_call.arguments == event2.tool_call.arguments
            return True

        # For ObservationEvents, compare the observation content, ignoring IDs
        if isinstance(event1, ObservationEvent) and isinstance(
            event2, ObservationEvent
        ):
            return (
                event1.source == event2.source
                and event1.observation == event2.observation
                and event1.tool_name == event2.tool_name
                # Ignore action_id, tool_call_id as they vary
            )

        # For AgentErrorEvents, compare the error content
        if isinstance(event1, AgentErrorEvent) and isinstance(event2, AgentErrorEvent):
            return (
                event1.source == event2.source and event1.error == event2.error
                # Ignore action_id as it varies
            )

        # For MessageEvents, compare the message content
        if isinstance(event1, MessageEvent) and isinstance(event2, MessageEvent):
            return (
                event1.source == event2.source
                and event1.llm_message == event2.llm_message
            )

        # Default fallback
        return event1 == event2

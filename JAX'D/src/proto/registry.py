from __future__ import annotations

from collections import defaultdict
import logging
from typing import Dict, List, Optional, Set

from .constraint_schema import AssumptionTag, PhysicsPredicate, StatusLevel

log = logging.getLogger("QUFT_Registry")


class PredicateRegistry:
    """Global registry for versioned predicates with dependency indexing."""

    def __init__(self):
        self._predicates: Dict[str, Dict[str, PhysicsPredicate]] = defaultdict(dict)
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._assumption_index: Dict[AssumptionTag, Set[str]] = defaultdict(set)

    def register(self, predicate: PhysicsPredicate) -> bool:
        if predicate.version in self._predicates[predicate.predicate_id]:
            log.warning("Duplicate registration: %s v%s", predicate.predicate_id, predicate.version)
            return False

        self._predicates[predicate.predicate_id][predicate.version] = predicate

        for dependency in predicate.dependencies:
            self._dependencies[dependency].add(predicate.predicate_id)

        for tag in predicate.assumptions:
            self._assumption_index[tag].add(predicate.predicate_id)

        log.info("Registered: %s v%s", predicate.predicate_id, predicate.version)
        return True

    def get_latest(self, predicate_id: str) -> Optional[PhysicsPredicate]:
        versions = self._predicates.get(predicate_id, {})
        if not versions:
            return None

        latest_version = max(versions, key=lambda version: tuple(int(x) for x in version.split(".")))
        return versions[latest_version]

    def propagate_assumption_failure(self, failed_tag: AssumptionTag) -> List[str]:
        affected_ids = list(self._assumption_index.get(failed_tag, set()))
        for predicate_id in affected_ids:
            predicate = self.get_latest(predicate_id)
            if predicate and predicate.status != StatusLevel.PENDING:
                log.warning("Assumption %s failed; %s should be downgraded to PENDING", failed_tag.value, predicate_id)
        return affected_ids

    def predicate_ids(self) -> List[str]:
        """Return a list of all registered predicate IDs."""
        return list(self._predicates.keys())

    def dependency_graph(self, predicate_id: str, depth: int = 3) -> Dict[str, List[str]]:
        graph: Dict[str, List[str]] = {}
        visited: Set[str] = set()

        def traverse(current_id: str, remaining: int):
            if remaining <= 0 or current_id in visited:
                return
            visited.add(current_id)
            dependents = sorted(self._dependencies.get(current_id, set()))
            graph[current_id] = dependents
            for dependent in dependents:
                traverse(dependent, remaining - 1)

        traverse(predicate_id, depth)
        return graph

"""Versionable solver and improvement policies with JSON serialization."""

from dataclasses import asdict, dataclass, field
import copy
import hashlib
import json

from .tasks import FAMILIES


@dataclass
class SolverSettings:
    nx: int = 32
    nt: int = 48
    grading: float = 1.0
    advection: str = "central"

    def validate(self):
        if type(self.nx) is not int or type(self.nt) is not int or not 8 <= self.nx <= 256 or not 8 <= self.nt <= 1024:
            raise ValueError("Allowed grid: 8<=nx<=256, 8<=nt<=1024")
        if not 1 <= self.grading <= 4 or self.advection not in ("central", "upwind"):
            raise ValueError("Invalid grading/advection policy")
        return self


@dataclass
class ImprovementPolicy:
    procedure: str = "uniform_refinement"
    priority: list[str] = field(default_factory=lambda: list(FAMILIES))
    refinement_factor: float = 2.0
    fractional_grading: float = 1.0

    def validate(self):
        if self.procedure not in ("uniform_refinement", "feedback_refinement"):
            raise ValueError("Unknown improvement procedure")
        if sorted(self.priority) != sorted(FAMILIES):
            raise ValueError("Priority must be a permutation of all families")
        if not 1.1 <= self.refinement_factor <= 2 or not 1 <= self.fractional_grading <= 4:
            raise ValueError("Invalid improvement parameters")
        return self


@dataclass
class AgentVersion:
    generation: int = 0
    parent_id: str | None = None
    solver: dict[str, SolverSettings] = field(default_factory=lambda: {f: SolverSettings() for f in FAMILIES})
    improvement: ImprovementPolicy = field(default_factory=ImprovementPolicy)
    memory: list[dict] = field(default_factory=list)
    foundation_model: str = "none: deterministic numerical scaffold"

    def to_dict(self):
        return asdict(self)

    @property
    def version_id(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, d):
        version = cls(generation=d["generation"], parent_id=d.get("parent_id"),
                      solver={f: SolverSettings(**settings).validate() for f, settings in d["solver"].items()},
                      improvement=ImprovementPolicy(**d["improvement"]).validate(),
                      memory=copy.deepcopy(d.get("memory", [])), foundation_model=d["foundation_model"])
        if set(version.solver) != set(FAMILIES):
            raise ValueError("Solver policy must cover all six families")
        return version


def propose(version, evolution_feedback, recursive):
    """A deterministic placeholder for model-written proposals; never reads sealed tasks.

    Fixed policy refines all families. Evolved policy prioritizes the largest relative
    error and chooses time grading for difficult fractional behavior. Both receive
    the same evaluation allowance. Selection uses independent development feedback.
    """
    candidate = copy.deepcopy(version)
    candidate.parent_id = version.version_id
    candidate.generation += 1
    policy = version.improvement
    selected = list(FAMILIES) if policy.procedure == "uniform_refinement" else policy.priority[:2]
    for family in selected:
        settings = candidate.solver[family]
        settings.nx = min(256, max(settings.nx+1, round(settings.nx*policy.refinement_factor)))
        settings.nt = min(1024, max(settings.nt+1, round(settings.nt*policy.refinement_factor)))
        if family in ("caputo", "multi_term"):
            settings.grading = policy.fractional_grading
    # Meta-policy is proposed alongside the solver and only survives accepted updates.
    # It is evaluated as a continuation policy, not claimed to work from this dev gain.
    if recursive:
        ranking = sorted(FAMILIES, key=lambda f: (-evolution_feedback["families"][f]["mean_relative_l2"], f))
        candidate.improvement = ImprovementPolicy(
            procedure="feedback_refinement", priority=ranking,
            refinement_factor=2.0, fractional_grading=2.0,
        )
    candidate.memory.append({"generation": version.generation, "evolution_feedback": evolution_feedback,
                             "selected_families": selected})
    return candidate

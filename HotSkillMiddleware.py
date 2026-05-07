from deepagents.middleware import SkillsMiddleware
from deepagents.middleware.skills import SkillMetadata, SkillsState, SkillsStateUpdate, _alist_skills_with_errors, \
    _list_skills_with_errors
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime


class HotSkillMiddleware(SkillsMiddleware):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def before_agent(self, state: SkillsState, runtime: Runtime, config: RunnableConfig) -> SkillsStateUpdate | None:  # ty: ignore[invalid-method-override]
        backend = self._get_backend(state, runtime, config)
        all_skills: dict[str, SkillMetadata] = {}
        skills_load_errors: list[str] = []

        # Load skills from each source in order
        # Later sources override earlier ones (last one wins)
        for source_path in self.sources:
            source_skills, source_error = _list_skills_with_errors(backend, source_path)
            if source_error is not None:
                skills_load_errors.append(source_error)
            for skill in source_skills:
                all_skills[skill["name"]] = skill

        skills = list(all_skills.values())
        update = SkillsStateUpdate(skills_metadata=skills)
        if skills_load_errors:
            update["skills_load_errors"] = skills_load_errors
        return update

    async def abefore_agent(self, state: SkillsState, runtime: Runtime, config: RunnableConfig) -> SkillsStateUpdate | None:  # ty: ignore[invalid-method-override]
        # Resolve backend (supports both direct instances and factory functions)
        backend = self._get_backend(state, runtime, config)
        all_skills: dict[str, SkillMetadata] = {}
        skills_load_errors: list[str] = []

        # Load skills from each source in order
        # Later sources override earlier ones (last one wins)
        for source_path in self.sources:
            source_skills, source_error = await _alist_skills_with_errors(backend, source_path)
            if source_error is not None:
                skills_load_errors.append(source_error)
            for skill in source_skills:
                all_skills[skill["name"]] = skill

        skills = list(all_skills.values())
        update = SkillsStateUpdate(skills_metadata=skills)
        if skills_load_errors:
            update["skills_load_errors"] = skills_load_errors
        return update
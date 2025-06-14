# Development Instructions

## Main Development Cycle

1. **Study specs/**
2. **Pick the highest value item from IMPLEMENTATION_PLAN.MD and implement it using up to 5-8 subagents.**
3. **Test the new changes then update IMPLEMENTATION_PLAN.MD to say the implementation is done using a single subagent.**
4. **Add changed code and IMPLEMENTATION_PLAN.MD with "git add -A" via bash then do a 'git commit' with a message that describes the changes you made**

## Code Quality

1. **Run automated tests and resolve test failures using a single subagent.**
2. **Important: when authoring documentation capture the why tests are important.**
3. **Important: We want single sources of truth, no migrations/adapters. If tests unrelated to your work fail then it's your job to resolve these tests as part of the increment of change.**
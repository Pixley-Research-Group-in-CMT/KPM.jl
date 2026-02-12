# Reusable Coding Instruction Playbook (Optional Dependency Migration)

## Goal
Migrate a hard dependency to optional extension mode with minimal risk.

## Step-by-step pattern
1. **Classify code paths**
   - Core/default path must compile and run without optional package.
   - Optional/accelerated path goes to extension (`ext/PackageExt.jl`).

2. **Move dependency metadata first**
   - Remove optional package from `[deps]`.
   - Add it to `[weakdeps]`.
   - Add `[extensions]` mapping.

3. **Make base code parse-safe**
   - Remove `using OptionalPkg` from base `src/` files.
   - Remove/replace optional-package-specific types/macros from always-loaded code.

4. **Create extension layer**
   - `module PackageExt`.
   - `using MainPackage` + optional dep(s).
   - Reintroduce specialized methods by extending `MainPackage.function_name`.

5. **Prefer small, behavior-preserving changes**
   - Keep function names and call sites stable.
   - Only specialize where needed.

6. **Add low-cost tests**
   - Default path tests that do not require optional runtime.
   - Verify fallback behavior and type stability basics.

7. **Self-review checklist**
   - Does base package load without optional dep?
   - Are macros/types from optional dep absent in base parse path?
   - Are extension methods namespaced correctly (`MainPackage.f`)?
   - Any method ambiguities introduced?

8. **If runtime unavailable**
   - Run static checks (grep, lint-like scans, file consistency).
   - Explicitly document unverified runtime risks.

## Commit strategy
- Commit A: metadata + compile-safety split.
- Commit B: extension reintroduction for specialized performance paths.
- Commit C: tests + docs + PR hygiene.

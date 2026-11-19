# Contributing to Play Store Review Analyzer

Thank you for your interest in contributing to Play Store Review Analyzer! We welcome bug reports, feature requests, documentation improvements, and code contributions.

## Development Workflow

1. **Fork & Branch:**
   - Fork the repository on GitHub.
   - Create a feature branch with a descriptive name:
     ```bash
     git checkout -b feat/your-feature-name
     # or: git checkout -b fix/issue-description
     ```

2. **Set Up Local Environment:**
   - Follow instructions in [README.md](README.md) to install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Copy `.env.example` to `.env` and configure local variables:
     ```bash
     cp .env.example .env
     ```

3. **Code Style & Formatting:**
   - Write clean, PEP 8 compliant Python code with clear docstrings and type annotations.
   - Ensure clean imports and modular structure across `src/`.

4. **Testing Requirements:**
   - Add unit or integration tests for new features and bug fixes in `tests/`.
   - Ensure all existing tests pass:
     ```bash
     python -m unittest discover -s tests
     ```

5. **Commit Message Guidelines:**
   - Write clear, concise commit messages following Conventional Commits format:
     - `feat: add user authentication endpoint`
     - `fix: resolve race condition in worker pool`
     - `docs: update quickstart instructions`

6. **Submitting a Pull Request:**
   - Push your branch to your fork.
   - Open a Pull Request targeting the `main` branch.
   - Fill in the PR description detailing your changes, verification steps, and referencing any related issues.

## Reporting Bugs

- Search existing issues to verify the bug has not already been reported.
- Open a new issue with a clear description, minimal reproducible example, and OS/runtime version details.

## Requesting Features

- Open a new issue with the feature description, use case, and proposed interface or API behavior.

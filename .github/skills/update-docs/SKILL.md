---
name: update-docs
description: "Update the documentation and the changelog."
---

## Workflow


1. First, identify the last release tag by running `git describe --tags --abbrev=0`. Use this tag as the baseline for all subsequent steps. If no tag exists, ask the user to provide the last release version manually before continuing.
2. Create a changelog based on the commits since the last release tag. If `git-changelog` is available, use it to generate a changelog from the commit history. If it is not available, fall back to running `git log <last-tag>..HEAD --oneline` and manually categorize the output. If there are no commits since the last release, inform the user and stop — do not proceed with changelog or documentation updates for an empty release. Categorize the changes appropriately (e.g., features, bug fixes, breaking changes).
3. Review the changelog generated in step 2. Then independently list all commits since the last release using `git log <last-tag>..HEAD` and compare them to the changelog to identify any missing entries or miscategorizations.
4. Search closed GitHub issues and pull requests that were closed or merged since the last release but not referenced in the generated changelog, and add links to those issues in the appropriate changelog entries.
5. Update the changelog as needed. Then print the changes to the chat and ask for approval to move forward with the release. If the user says no or requests changes, ask them to specify what should be changed, apply those changes, and re-present the updated changelog for approval before proceeding. If the answer is yes, proceed to the next step.
6. Add the changelog to the `CHANGELOG.md` file in the `mkdocs/docs` folder of the repository. Follow the format already used in `CHANGELOG.md`. If the file is empty or does not exist, use the Keep a Changelog format (https://keepachangelog.com) with sections: Added, Changed, Deprecated, Removed, Fixed, Security. Include the version number and release date at the top of the new section.
7. Now, review the documentation throughout the `mkdocs/docs` folder to ensure any version-specific information is updated for the new version. This includes updating version numbers, dates, and any other relevant details. In case new features are implemented, make sure to add documentation for those features as well into the appropriate sections. In case no sections exists, please create a new section for the new feature and add the documentation there. Make sure to format it properly and include any relevant examples or usage instructions.
8. After updating the documentation, commit the changes to the repository with a clear commit message indicating that the documentation and changelog have been updated for the new release. For example, "Update documentation and changelog for version X.Y.Z release." Commit on the current branch. Do not push or create a pull request unless explicitly instructed. If the repository requires pull requests for the main branch, note this to the user and stop.

# claude-harness-study

this repo for reproducing claude harness

## Local configuration

Copy `.env.example` to `.env`, then fill in the API key, optional API base URL,
and model ID locally. Keep real credentials only in your local `.env` file.
The example contains no credentials, and `.env` is ignored by Git.

Adding `.gitignore` does not remove credentials from earlier commits.
Revoke/rotate any API key that has already been exposed.

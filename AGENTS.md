Stack migration complete. Frontend is React (Vite) + FastAPI. Old Streamlit application and dependencies have been removed. Keep src/ui_utils.py (proxying to src/core/) as the source of truth for all inference and benchmark loading logic.

### Dual-Remote Git Policy
- `origin`: `https://github.com/siddharthsingh18/IFSI-for-phishing-email-detection.git` (Collaborator repository)
- `myrepo`: `https://github.com/Sanchitk99/IFSI-phishing-email-detection.git` (Personal repository)
- Local `main` tracks `myrepo/main`.
- Never rename or remove either remote.
- Never use `git push --force` or `-f` without explicit user permission.
- Always check branch and working tree before pushing.
- When synchronizing changes upon request, push to both remotes:
  1. `git push origin main`
  2. `git push myrepo main`


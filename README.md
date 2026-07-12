# fontem-consolidator

Entity-resolution + rule engine. Consumes events.entity_events, runs rules (LEI-match, fuzzy-name, successor, name+country, …), and either auto-merges duplicates or flags SAME_AS candidates for human review. Hosts its own /resolve and /consolidate HTTP API plus the consolidator-trigger event consumer.

## Deploy

CI auto-deploys to the testing env on every merge to main. Promotion to staging / prod is **manual** — bump the version in `gitops/<env>/<service>.yaml` to land it in a given environment.

## Convention

See [/config/repos/CLAUDE.md](https://contribute.void42.internal/fontem/gitops) for workspace-wide rules (feature branches + CI gate, no direct push to main, full gate before declaring done, conventional commits).

<!-- rebuild-trigger: cosign v3 with .sig-legacy fix -->

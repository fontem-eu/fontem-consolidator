> ### 🪞 This GitHub repository is a mirror
>
> Development happens on Fontem's own infrastructure; this mirror is
> updated automatically. **Issues and pull requests opened here are not
> monitored.**
>
> If you would like to contribute — code, data sources, review, or
> anything else — please get in touch at **team@fontem.eu** and we will
> set you up.

# fontem-consolidator

Entity-resolution + rule engine. Consumes events.entity_events, runs rules (LEI-match, fuzzy-name, successor, name+country, …), and either auto-merges duplicates or records a :SAME_AS_CANDIDATE for human review. A candidate asserts nothing: only an approved equivalence becomes a :SAME_AS edge and an AssertSameAs event, and a :NOT_SAME_AS withdraws one that turned out wrong. Hosts its own /resolve and /consolidate HTTP API plus the consolidator-trigger event consumer.

## Deploy

CI auto-deploys to the testing env on every merge to main. Promotion to staging / prod is **manual** — bump the version in `gitops/<env>/<service>.yaml` to land it in a given environment.

## Convention

See [/config/repos/CLAUDE.md](https://contribute.void42.internal/fontem/gitops) for workspace-wide rules (feature branches + CI gate, no direct push to main, full gate before declaring done, conventional commits).

<!-- rebuild-trigger: cosign v3 with .sig-legacy fix -->

## License

Apache License 2.0 — see [LICENSE](LICENSE).

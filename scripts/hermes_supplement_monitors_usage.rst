Supplement Hermes monitor results from saved trajectories
==========================================================

After a Hermes all-monitor batch finishes, use ``supplement_hermes_monitors.sh``
to replay selected monitors on its saved ``guard_lifecycle.jsonl``. This does
not rerun Hermes, tools, or the benchmark evaluator. It can make new API calls
to the selected monitor or its rule generator. It needs the same API credentials,
local model services, policy/model files, and prebuilt Docker base image.

By default, only runs whose current label for a selected method is ``error``
are retried. Use ``--all-runs`` when a monitor implementation changed and every
label needs recomputation. Preview the exact phases first::

    ./scripts/supplement_hermes_monitors.sh \
      jobs/agentdojo_user_task_15_injection_5/Hermes/gpt/all_monitors \
      --methods aegis llamafirewall --dry-run

    ./scripts/supplement_hermes_monitors.sh \
      jobs/agentdojo_user_task_15_injection_5/Hermes/gpt/all_monitors \
      --methods aegis llamafirewall

For AgentDojo, the ``all_monitors`` path picks the latest batch name present in
both injection conditions. Pass ``--batch NAME`` for an older paired batch, or
pass one exact batch directory to process only that condition. Use repeatable
``--run 3`` to select run indices.

ASB and PrivacyLens accept an exact batch directory::

    ./scripts/supplement_hermes_monitors.sh \
      jobs/asb_all_monitors/CASE/BATCH \
      --methods agentdog --dry-run

    ./scripts/supplement_hermes_monitors.sh \
      jobs/privacylens_live_all_monitors/CASE/BATCH \
      --methods toolsafe --all-runs --dry-run

ASB replays both ``target`` and ``control`` when their saved artifacts exist;
PrivacyLens replays only ``target``. An ``ok: false`` summary entry can still
be processed when its phase has a lifecycle, replay manifest, saved config,
and evaluation file.

The helper finds the newest local ``<container.image>-base-*`` Docker image.
Use ``--image EXACT_TAG`` to pin another compatible image. The per-method
timeout defaults to 1800 seconds; change it with ``--timeout SECONDS``. For
older saved configs containing removed Pro2Guard fields, a migrated YAML copy
is written only in the attempt; its original config is preserved. This replays
under the current plugin implementation, so compare results with that version
in mind.

Each attempt is written under
``run_###/PHASE/defense_supplements/TIMESTAMP_ID/METHOD/``, including
``attempt.json`` and container stdout/stderr. Policy-generator batch caches
and AGrail memory are copied into the attempt before replay, so replay writes
stay isolated. A valid replay is activated through that phase's
``defense_supplements.json``. Original replay, manifest, and evaluation files
remain unchanged. Failed or decisionless attempts stay on disk for diagnosis
but are not activated. After activation, the appropriate analyzer runs
and reads the supplement. Previous analysis files are copied to
``analysis/before_supplement/TIMESTAMP_ID/``. ``run_labels.csv`` marks
activated rows with ``supplemented=True``.

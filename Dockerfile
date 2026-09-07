FROM nousresearch/hermes-agent:v2026.8.31@sha256:64923faeae267792bf9bf87fe3b4c4869e35004e360c7df01730ad801b74d524

LABEL org.opencontainers.image.source="https://github.com/chigwell/hermes-penelopa" \
      io.penelopa.hermes.revision="29112bef099274229cadff79cdff7bf7b99c4b77"

USER root
RUN mkdir -p /opt/penelopa /opt/data/hermes /run/penelopa \
    && chown 10001:10001 /opt/data /opt/data/hermes /run/penelopa \
    && HERMES_HOME=/tmp/hermes-build-contract /opt/hermes/.venv/bin/python -c "from run_agent import AIAgent; from hermes_cli.goals import GoalManager; from agent.background_review import spawn_background_review_thread"
COPY penelopa_runtime /opt/penelopa/penelopa_runtime
COPY tests/test_native_contract.py /opt/penelopa/tests/test_native_contract.py
ENV PYTHONPATH=/opt/penelopa:/opt/hermes \
    HERMES_HOME=/opt/data/hermes \
    HERMES_WRITE_SAFE_ROOT=/opt/data/hermes \
    HERMES_DISABLE_LAZY_INSTALLS=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /opt/penelopa
ENTRYPOINT ["/opt/hermes/.venv/bin/python", "-m", "penelopa_runtime"]

# Builds the `vesuvius-render` CLI for use as an Argo workflow image (vesuvius-atlas
# `vesuvius-render` WorkflowTemplate). Built natively per-arch (kaniko on a c8gd arm64 node,
# or buildx for amd64); both linux/amd64 and linux/arm64 are supported.
#
# Per-target target-cpu tuning: cargo reads the CARGO_TARGET_<TRIPLE>_RUSTFLAGS matching the
# host triple being built, so both can be set unconditionally and the right one applies:
#   - x86_64  : x86-64-v3  (Haswell baseline: AVX2/BMI2/FMA)
#   - aarch64 : neoverse-v2 (Graviton4 / c8g)
#
# `cargo build -p vesuvius-render` only compiles the render binary's dependency subtree
# (vesuvius-rs, vesuvius-zarr) — the eframe/egui GUI crate is not pulled in, so no GUI
# system libraries are needed. The build has no native deps beyond a C compiler (for the
# lz4/zstd `cc` build deps), which `rust:bookworm` already provides.

FROM rust:1-bookworm AS builder
ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS="--cfg tokio_unstable -C target-cpu=x86-64-v3"
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="--cfg tokio_unstable -C target-cpu=neoverse-v2"
WORKDIR /src
COPY . .
RUN cargo build --release -p vesuvius-render

FROM debian:bookworm-slim
# ca-certificates: HTTPS volume/overlay fetches. awscli: copy renders to S3 in the workflow.
# `uname -m` is x86_64 / aarch64, matching the awscli download names directly.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl unzip; \
    rm -rf /var/lib/apt/lists/*; \
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" -o awscliv2.zip; \
    unzip -q awscliv2.zip; ./aws/install; rm -rf awscliv2.zip aws

COPY --from=builder /src/target/release/vesuvius-render /usr/local/bin/vesuvius-render

ENTRYPOINT []
CMD ["vesuvius-render", "--help"]

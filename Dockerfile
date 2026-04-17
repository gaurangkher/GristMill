# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Build Rust daemon
# ─────────────────────────────────────────────────────────────────────────────
# Pin to Bookworm so glibc matches node:20-slim (also Bookworm / glibc 2.36).
# rust:latest recently moved to Trixie which breaks libstdc++/libmvec compat.
FROM rust:1-bookworm AS rust-builder

# Install C/C++ build tools needed by crates with native dependencies (usearch, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY gristmill-core/ ./gristmill-core/

RUN cd gristmill-core && \
    cargo build --release -p gristmill-daemon

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Build TypeScript shell
# ─────────────────────────────────────────────────────────────────────────────
FROM node:20-bookworm-slim AS ts-builder

RUN npm install -g pnpm@9

WORKDIR /build
COPY gristmill-integrations/package.json \
     gristmill-integrations/pnpm-lock.yaml \
     gristmill-integrations/tsconfig.json ./

RUN pnpm install --frozen-lockfile

COPY gristmill-integrations/src/ ./src/

# Build the React SPA (outputs to src/dashboard/ui/dist/)
RUN cd src/dashboard/ui && npm install && npm run build

# Compile the TypeScript server (outputs to dist/)
RUN pnpm build

# Place the React dist where the compiled server expects it:
#   _resolveUiDist() resolves to dist/dashboard/ui/dist/ relative to dist/
RUN mkdir -p dist/dashboard/ui && cp -r src/dashboard/ui/dist dist/dashboard/ui/

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Final runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM node:20-bookworm-slim AS runtime

ARG BUILD_DATE
ARG GIT_SHA

LABEL org.opencontainers.image.title="GristMill"
LABEL org.opencontainers.image.description="Local-first AI orchestration: Rust grinds, Python trains, TypeScript connects"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${GIT_SHA}"

# Install only runtime dependencies
# gosu is used by the entrypoint to drop from root → gristmill user after
# fixing named-volume permissions on /gristmill/run.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gosu \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g pnpm@9

# Create non-root user
RUN groupadd -r gristmill && useradd -r -g gristmill -d /app gristmill

WORKDIR /app

# Copy Rust daemon
COPY --from=rust-builder \
    /build/gristmill-core/target/release/gristmill-daemon \
    /usr/local/bin/gristmill-daemon

# Copy C++ runtime libs from the builder so CXXABI version and libmvec match.
# node:20-slim strips libstdc++ and libmvec; the builder (rust:latest) has the
# full set. We resolve the arch at build time so this works on amd64 and arm64.
RUN --mount=type=bind,from=rust-builder,source=/,target=/builder \
    sh -c 'ARCH=$(uname -m)-linux-gnu; \
           cp /builder/usr/lib/$ARCH/libstdc++.so.6* /usr/lib/$ARCH/ 2>/dev/null || true; \
           cp /builder/usr/lib/$ARCH/libmvec*        /usr/lib/$ARCH/ 2>/dev/null || true; \
           ldconfig'

# Copy TypeScript shell build + production dependencies
COPY --from=ts-builder /build/dist ./dist
COPY --from=ts-builder /build/node_modules ./node_modules
COPY gristmill-integrations/package.json ./

# Runtime directories
RUN mkdir -p /data/gristmill/feedback \
             /data/gristmill/models \
             /data/gristmill/memory \
             /data/gristmill/checkpoints \
             /data/gristmill/plugins \
             /data/gristmill/db \
    && chown -R gristmill:gristmill /app /data

# NOTE: We intentionally do NOT set USER here.
# The entrypoint runs as root, fixes named-volume permissions on /gristmill/run,
# then uses gosu to drop to the gristmill user for both the daemon and the TS shell.

# Config via environment or mounted file
ENV GRISTMILL_CONFIG=/data/gristmill/config.yaml
ENV GRISTMILL_SOCK=/data/gristmill/gristmill.sock
ENV NODE_ENV=production
ENV PORT=3000
ENV HOST=0.0.0.0

EXPOSE 3000

# Healthcheck hits the dashboard API
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000/api/metrics/health', r => process.exit(r.statusCode === 200 ? 0 : 1)).on('error', () => process.exit(1))"

# Entrypoint starts Rust daemon then TypeScript shell
COPY docker-entrypoint.sh /usr/local/bin/
ENTRYPOINT ["docker-entrypoint.sh"]

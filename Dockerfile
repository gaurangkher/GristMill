# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Build Rust daemon (static musl binary)
# ─────────────────────────────────────────────────────────────────────────────
FROM rust:1.80-slim AS rust-builder

RUN apt-get update && apt-get install -y musl-tools && rm -rf /var/lib/apt/lists/*
RUN rustup target add x86_64-unknown-linux-musl

WORKDIR /build
COPY gristmill-core/ ./gristmill-core/

RUN cd gristmill-core && \
    cargo build --release --target x86_64-unknown-linux-musl -p gristmill-daemon

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Build TypeScript shell
# ─────────────────────────────────────────────────────────────────────────────
FROM node:20-slim AS ts-builder

RUN npm install -g pnpm@9

WORKDIR /build
COPY gristmill-integrations/package.json \
     gristmill-integrations/pnpm-lock.yaml \
     gristmill-integrations/tsconfig.json ./

RUN pnpm install --frozen-lockfile

COPY gristmill-integrations/src/ ./src/
RUN pnpm build

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Final runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM node:20-slim AS runtime

ARG BUILD_DATE
ARG GIT_SHA

LABEL org.opencontainers.image.title="GristMill"
LABEL org.opencontainers.image.description="Local-first AI orchestration: Rust grinds, Python trains, TypeScript connects"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${GIT_SHA}"

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g pnpm@9

# Create non-root user
RUN groupadd -r gristmill && useradd -r -g gristmill -d /app gristmill

WORKDIR /app

# Copy Rust daemon
COPY --from=rust-builder \
    /build/gristmill-core/target/x86_64-unknown-linux-musl/release/gristmill-daemon \
    /usr/local/bin/gristmill-daemon

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
    && chown -R gristmill:gristmill /app /data

USER gristmill

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

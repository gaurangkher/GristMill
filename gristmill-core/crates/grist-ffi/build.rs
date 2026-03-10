// build.rs — required by napi-rs to generate the Node.js module registration
// boilerplate.  Only active when the `node` feature is enabled, but
// napi-build silently no-ops when `CARGO_FEATURE_NODE` is absent.

#[cfg(feature = "node")]
extern crate napi_build;

fn main() {
    #[cfg(feature = "node")]
    napi_build::setup();
}

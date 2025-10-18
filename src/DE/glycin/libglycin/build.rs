fn main() {
    let major = std::env::var("CARGO_PKG_VERSION_MAJOR").unwrap();
    // Set soname of library
    println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libglycin-{major}.so.0");
}

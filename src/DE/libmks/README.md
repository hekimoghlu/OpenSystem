# libmks

This library provides a "Mouse, Keyboard, and Screen" to QEMU using the
D-Bus device support in QEMU and GTK 4.

# Documentation

Nightly documentation can be found [here](https://gnome.pages.gitlab.gnome.org/libmks/libmks1).

# Unit testing

Be sure you have `lcov` package installed on your system if you want coverage data.

```bash
meson setup builddir
meson configure -Db_coverage=true builddir  # if you want to collect coverage data
meson compile -C builddir
meson test -C builddir --suit "libmks"
rm -rf builddir/subprojects  # if you don't want subprojects coverage in the report
ninja coverage-html -C builddir  # if you want to generate coverage report
```

If generated, coverage report will be in `builddir/meson-logs/coveragereport/index.html`

# Testing

By default, QEMU will connect to your user session D-Bus if you do not
provide an address for `-display dbus`. Therefore, it is pretty easy to
test things by running QEMU manually and then connecting with the test
program `./tools/mks`.

```sh
qemu-img create -f qcow2 fedora.img 30G
qemu-system-x86_64 \
    -enable-kvm \
    -cpu host \
    -device virtio-vga-gl,xres=1920,yres=1080 \
    -m 8G \
    -smp 4 \
    -display dbus,gl=on \
    -cdrom Fedora-Workstation.iso \
    -hda fedora.img \
    -boot d
```

and then to run the test widget

```sh
meson setup build
cd build
ninja
./tools/mks
```

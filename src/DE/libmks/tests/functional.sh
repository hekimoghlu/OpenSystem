#!/usr/bin/bash

test -e 512-byte-vm.raw || wget https://github.com/oVirt/512-byte-vm/releases/download/2.0.0/512-byte-vm.raw

QEMU=""

ARCH="$(uname -m)"
test -x "/usr/bin/qemu-system-$ARCH" && QEMU="/usr/bin/qemu-system-$ARCH"
test -x /usr/libexec/qemu-kvm && QEMU=/usr/libexec/qemu-kvm

if [ -z "$QEMU" ]
then
    echo "Missing QEMU executable"
    exit 1
fi

echo -e "Using $QEMU\n"

${QEMU} \
       -drive file=512-byte-vm.raw,format=raw \
       -display dbus -device virtio-vga \
       -serial mon:stdio \
       -enable-kvm &

QEMUPID=$!

sleep 1
BUILDDIR=${BUILDDIR:="builddir"}
"../${BUILDDIR}/tools/mks-connect"
"../${BUILDDIR}/tools/mks" &
MKSPID=$!

sleep 1

kill -SIGTERM $QEMUPID
kill -SIGTERM $MKSPID

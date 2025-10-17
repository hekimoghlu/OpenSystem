/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 17, 2025.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include "config.h"
#include <wtf/glib/Sandbox.h>

#include <gio/gio.h>
#include <wtf/FileSystem.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/text/CString.h>

namespace WTF {

bool isInsideFlatpak()
{
    static bool returnValue = g_file_test("/.flatpak-info", G_FILE_TEST_EXISTS);
    return returnValue;
}

#if ENABLE(BUBBLEWRAP_SANDBOX)
bool isInsideUnsupportedContainer()
{
    static bool inContainer = g_file_test("/run/.containerenv", G_FILE_TEST_EXISTS);
    static int supportedContainer = -1;

    // Being in a container does not mean sub-containers cannot work. It depends upon various details such as
    // docker vs podman, which permissions are given, is it privileged or unprivileged, and are unprivileged user namespaces enabled.
    // So this just does a basic test of if `bwrap` runs successfully.
    if (inContainer && supportedContainer == -1) {
        const char* bwrapArgs[] = {
            BWRAP_EXECUTABLE,
            "--ro-bind", "/", "/",
            "--proc", "/proc",
            "--dev", "/dev",
            "--unshare-all",
            "true",
            nullptr
        };
        int waitStatus = 0;
        gboolean spawnSucceeded = g_spawn_sync(nullptr, const_cast<char**>(bwrapArgs), nullptr,
            G_SPAWN_STDERR_TO_DEV_NULL, nullptr, nullptr, nullptr, nullptr, &waitStatus, nullptr);
        supportedContainer = spawnSucceeded && g_spawn_check_exit_status(waitStatus, nullptr);
        if (!supportedContainer)
            WTFLogAlways("Bubblewrap does not work inside of this container, sandboxing will be disabled.");
    }

    return inContainer && !supportedContainer;
}
#endif

bool isInsideSnap()
{
    // The "SNAP" environment variable is not unlikely to be set for/by something other
    // than Snap, so check a couple of additional variables to avoid false positives.
    // See: https://snapcraft.io/docs/environment-variables
    static bool returnValue = g_getenv("SNAP") && g_getenv("SNAP_NAME") && g_getenv("SNAP_REVISION");
    return returnValue;
}

bool shouldUseBubblewrap()
{
#if ENABLE(BUBBLEWRAP_SANDBOX)
    return !isInsideFlatpak() && !isInsideSnap() && !isInsideUnsupportedContainer();
#else
    return false;
#endif
}

bool shouldUsePortal()
{
    static bool returnValue = []() -> bool {
        const char* usePortal = isInsideFlatpak() || isInsideSnap() ? "1" : g_getenv("WEBKIT_USE_PORTAL");
        return usePortal && usePortal[0] != '0';
    }();
    return returnValue;
}

bool checkFlatpakPortalVersion(int version)
{
    static int flatpakPortalVersion = -1;
    static std::once_flag onceFlag;

    std::call_once(onceFlag, [] {
        GRefPtr<GDBusProxy> proxy = adoptGRef(g_dbus_proxy_new_for_bus_sync(G_BUS_TYPE_SESSION, G_DBUS_PROXY_FLAGS_NONE, nullptr, "org.freedesktop.portal.Flatpak", "/org/freedesktop/portal/Flatpak", "org.freedesktop.portal.Flatpak", nullptr, nullptr));
        if (!proxy)
            return;
        GRefPtr<GVariant> result = adoptGRef(g_dbus_proxy_get_cached_property(proxy.get(), "version"));
        if (!result)
            return;
        flatpakPortalVersion = g_variant_get_uint32(result.get());
    });

    return flatpakPortalVersion != -1 && flatpakPortalVersion >= version;
}

const CString& sandboxedUserRuntimeDirectory()
{
    static LazyNeverDestroyed<CString> userRuntimeDirectory;
    static std::once_flag onceKey;
    std::call_once(onceKey, [] {
#if PLATFORM(GTK)
        static constexpr ASCIILiteral baseDirectory = "webkitgtk"_s;
#elif PLATFORM(WPE)
        static constexpr ASCIILiteral baseDirectory = "wpe"_s;
#else
        static constexpr ASCIILiteral baseDirectory = "javascriptcore"_s;
#endif
        userRuntimeDirectory.construct(FileSystem::pathByAppendingComponent(FileSystem::stringFromFileSystemRepresentation(g_get_user_runtime_dir()), baseDirectory).utf8());
    });
    return userRuntimeDirectory.get();
}

} // namespace WTF

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 1, 2021.
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
#include "FlatpakLauncher.h"

#if OS(LINUX)

#include <gio/gio.h>
#include <wtf/glib/GUniquePtr.h>
#include <wtf/glib/Sandbox.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // GTK/WPE port

namespace WebKit {

GRefPtr<GSubprocess> flatpakSpawn(GSubprocessLauncher* launcher, const WebKit::ProcessLauncher::LaunchOptions& launchOptions, char** argv, int childProcessSocket, int pidSocket, GError** error)
{
    ASSERT(launcher);

    // When we are running inside of flatpak's sandbox we do not have permissions to use the same
    // bubblewrap sandbox we do outside but flatpak offers the ability to create new sandboxes
    // for us using flatpak-spawn.

    GUniquePtr<char> childProcessSocketArg(g_strdup_printf("--forward-fd=%d", childProcessSocket));
    GUniquePtr<char> pidSocketArg(g_strdup_printf("--forward-fd=%d", pidSocket));
    Vector<CString> flatpakArgs = {
        "flatpak-spawn",
        childProcessSocketArg.get(),
        pidSocketArg.get(),
        "--expose-pids",
        "--watch-bus"
    };

    if (launchOptions.processType == ProcessLauncher::ProcessType::Web) {
        flatpakArgs.appendVector(Vector<CString>({
            "--sandbox",
            "--no-network",
            "--sandbox-flag=share-gpu",
            "--sandbox-flag=share-display",
            "--sandbox-flag=share-sound",
            "--sandbox-flag=allow-a11y",
            "--sandbox-flag=allow-dbus", // Note that this only allows portals and $appid.Sandbox.* access
        }));

        for (const auto& pathAndPermission : launchOptions.extraSandboxPaths) {
            const char* formatString = pathAndPermission.value == SandboxPermission::ReadOnly ? "--sandbox-expose-path-ro=%s": "--sandbox-expose-path=%s";
            GUniquePtr<gchar> pathArg(g_strdup_printf(formatString, pathAndPermission.key.data()));
            flatpakArgs.append(pathArg.get());
        }

#if USE(ATSPI)
        RELEASE_ASSERT(isInsideFlatpak());
        if (checkFlatpakPortalVersion(7)) {
            auto busName = launchOptions.extraInitializationData.get<HashTranslatorASCIILiteral>("accessibilityBusName"_s);
            GUniquePtr<gchar> a11yOwnNameArg(g_strdup_printf("--sandbox-a11y-own-name=%s", busName.utf8().data()));
            flatpakArgs.append(a11yOwnNameArg.get());
        }
#endif
    }

    // We need to pass our full environment to the subprocess.
    GUniquePtr<char*> environ(g_get_environ());
    for (char** variable = environ.get(); variable && *variable; variable++) {
        GUniquePtr<char> arg(g_strconcat("--env=", *variable, nullptr));
        flatpakArgs.append(arg.get());
    }

    char** newArgv = g_newa(char*, g_strv_length(argv) + flatpakArgs.size() + 1);
    size_t i = 0;

    for (const auto& arg : flatpakArgs)
        newArgv[i++] = const_cast<char*>(arg.data());
    for (size_t x = 0; argv[x]; x++)
        newArgv[i++] = argv[x];
    newArgv[i++] = nullptr;

    return adoptGRef(g_subprocess_launcher_spawnv(launcher, newArgv, error));
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

};

#endif // OS(LINUX)

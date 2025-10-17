/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 26, 2024.
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
#include "RemoteInspectorUtils.h"

#if ENABLE(REMOTE_INSPECTOR)

#include <gio/gio.h>
#include <mutex>
#include <wtf/SHA1.h>
#include <wtf/glib/GSpanExtras.h>

#define INSPECTOR_BACKEND_COMMANDS_PATH "/org/webkit/inspector/UserInterface/Protocol/InspectorBackendCommands.js"

namespace Inspector {

GRefPtr<GBytes> backendCommands()
{
#if PLATFORM(WPE)
    static std::once_flag flag;
    std::call_once(flag, [] {
        const char* dataDir = PKGDATADIR;
        GUniqueOutPtr<GError> error;

        const char* path = g_getenv("WEBKIT_INSPECTOR_RESOURCES_PATH");
        if (path && g_file_test(path, G_FILE_TEST_IS_DIR))
            dataDir = path;

        GUniquePtr<char> gResourceFilename(g_build_filename(dataDir, "inspector.gresource", nullptr));
        GRefPtr<GResource> gresource = adoptGRef(g_resource_load(gResourceFilename.get(), &error.outPtr()));
        if (!gresource) {
            g_error("Error loading inspector.gresource: %s", error->message);
        }
        g_resources_register(gresource.get());
    });
#endif
    GRefPtr<GBytes> bytes = adoptGRef(g_resources_lookup_data(INSPECTOR_BACKEND_COMMANDS_PATH, G_RESOURCE_LOOKUP_FLAGS_NONE, nullptr));
    ASSERT(bytes);
    return bytes;
}

const CString& backendCommandsHash()
{
    static CString hexDigest;
    if (hexDigest.isNull()) {
        auto bytes = backendCommands();
        auto bytesSpan = span(bytes);
        ASSERT(bytesSpan.size());
        SHA1 sha1;
        sha1.addBytes(bytesSpan);
        hexDigest = sha1.computeHexDigest();
    }
    return hexDigest;
}

} // namespace Inspector

#endif // ENABLE(REMOTE_INSPECTOR)

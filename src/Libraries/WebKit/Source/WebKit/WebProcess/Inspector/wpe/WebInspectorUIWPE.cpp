/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 2, 2024.
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
#include "WebInspectorUI.h"

#if ENABLE(WPE_PLATFORM)

#include <gio/gio.h>
#include <wtf/glib/GUniquePtr.h>

namespace WebKit {

void WebInspectorUI::didEstablishConnection()
{
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        const char* dataDir = PKGDATADIR;
        GUniqueOutPtr<GError> error;

#if ENABLE(DEVELOPER_MODE)
        const char* path = g_getenv("WEBKIT_INSPECTOR_RESOURCES_PATH");
        if (path && g_file_test(path, G_FILE_TEST_IS_DIR))
            dataDir = path;
#endif
        GUniquePtr<char> gResourceFilename(g_build_filename(dataDir, "inspector.gresource", nullptr));
        GRefPtr<GResource> gresource = adoptGRef(g_resource_load(gResourceFilename.get(), &error.outPtr()));
        if (!gresource) {
            g_error("Error loading inspector.gresource: %s", error->message);
        }
        g_resources_register(gresource.get());
    });
}

bool WebInspectorUI::canSave(InspectorFrontendClient::SaveMode)
{
    return false;
}

bool WebInspectorUI::canLoad()
{
    return false;
}

bool WebInspectorUI::canPickColorFromScreen()
{
    return false;
}

String WebInspectorUI::localizedStringsURL() const
{
    return "resource:///org/webkit/inspector/Localizations/en.lproj/localizedStrings.js"_s;
}

} // namespace WebKit

#endif // ENABLE(WPE_PLATFORM)

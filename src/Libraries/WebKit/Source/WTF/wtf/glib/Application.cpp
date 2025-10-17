/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 7, 2023.
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
#include "Application.h"

#if USE(GLIB)
#include <glib.h>
#include <mutex>
#include <wtf/FileSystem.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/UUID.h>
#include <wtf/glib/GUniquePtr.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>

namespace WTF {

const CString& applicationID()
{
    static NeverDestroyed<CString> id;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        if (auto* app = g_application_get_default()) {
            if (const char* appID = g_application_get_application_id(app)) {
                id.get() = appID;
                return;
            }
        }

        const char* programName = g_get_prgname();
        if (programName && g_application_id_is_valid(programName)) {
            id.get() = programName;
            return;
        }

        // There must be some id for xdg-desktop-portal to function.
        // xdg-desktop-portal uses this id for permissions.
        // This creates a somewhat reliable id based on the executable path
        // which will avoid potentially gaining permissions from another app
        // and won't flood xdg-desktop-portal with new ids.
        if (auto executablePath = FileSystem::currentExecutablePath(); !executablePath.isNull()) {
            GUniquePtr<char> digest(g_compute_checksum_for_data(G_CHECKSUM_SHA256, reinterpret_cast<const uint8_t*>(executablePath.data()), executablePath.length()));
            id.get() = makeString("org.webkit.app-"_s, unsafeSpan(digest.get())).utf8();
            return;
        }

        // If it is not possible to obtain the executable path, generate a random identifier as a fallback.
        auto uuid = WTF::UUID::createVersion4Weak();
        id.get() = makeString("org.webkit.app-"_s, uuid.toString()).utf8();
    });
    return id.get();
}

} // namespace WTF

#endif // USE(GLIB)

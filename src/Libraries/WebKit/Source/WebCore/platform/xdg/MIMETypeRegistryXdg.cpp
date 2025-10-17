/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 31, 2025.
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
#include "MIMETypeRegistry.h"

#include <wtf/text/MakeString.h>

#define XDG_PREFIX _wk_xdg
#include "xdgmime.h"

#define MAX_EXTENSION_COUNT 10
namespace WebCore {

String MIMETypeRegistry::mimeTypeForExtension(StringView string)
{
    if (string.isEmpty())
        return String();

    // Build any filename with the given extension.
    auto filename = makeString("a."_s, string);
    if (const char* mimeType = xdg_mime_get_mime_type_from_file_name(filename.utf8().data())) {
        if (mimeType != XDG_MIME_TYPE_UNKNOWN)
            return String::fromUTF8(mimeType);
    }

    return String();
}

bool MIMETypeRegistry::isApplicationPluginMIMEType(const String&)
{
    return false;
}

String MIMETypeRegistry::preferredExtensionForMIMEType(const String& mimeType)
{
    if (mimeType.isEmpty())
        return String();

    if (mimeType.startsWith("text/plain"_s))
        return String();

    String returnValue;
    char* extension;
    if (xdg_mime_get_simple_globs(mimeType.utf8().data(), &extension, 1)) {
        auto view = std::string_view(extension);
        if (view[0] == '.' && view.size() > 1)
            returnValue = String::fromUTF8(view.substr(1).data());
        free(extension);
    }
    return returnValue;
}

Vector<String> MIMETypeRegistry::extensionsForMIMEType(const String& mimeType)
{
    if (mimeType.isEmpty())
        return { };

    Vector<String> returnValue;
    std::array<char*, MAX_EXTENSION_COUNT> extensions;
    int n = xdg_mime_get_simple_globs(mimeType.utf8().data(), extensions.data(), MAX_EXTENSION_COUNT);
    for (int i = 0; i < n; ++i) {
        returnValue.append(String::fromUTF8(extensions[i]));
        free(extensions[i]);
    }
    return returnValue;
}

}

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 8, 2025.
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

namespace WebCore {

static const std::initializer_list<TypeExtensionPair>& platformMediaTypes()
{
    static std::initializer_list<TypeExtensionPair> platformMediaTypes = {
        { "image/bmp"_s, "bmp"_s },
        { "text/css"_s, "css"_s },
        { "image/gif"_s, "gif"_s },
        { "text/html"_s, "htm"_s },
        { "text/html"_s, "html"_s },
        { "image/x-icon"_s, "ico"_s },
        { "image/jpeg"_s, "jpeg"_s },
        { "image/jpeg"_s, "jpg"_s },
        { "application/x-javascript"_s, "js"_s },
        { "application/pdf"_s, "pdf"_s },
        { "image/png"_s, "png"_s },
        { "application/rss+xml"_s, "rss"_s },
        { "image/svg+xml"_s, "svg"_s },
        { "application/x-shockwave-flash"_s, "swf"_s },
        { "text/plain"_s, "text"_s },
        { "text/plain"_s, "txt"_s },
        { "text/vnd.wap.wml"_s, "wml"_s },
        { "application/vnd.wap.wmlc"_s, "wmlc"_s },
        { "image/x-xbitmap"_s, "xbm"_s },
        { "application/xhtml+xml"_s, "xhtml"_s },
        { "text/xml"_s, "xml"_s },
        { "text/xsl"_s, "xsl"_s },
    };
    return platformMediaTypes;
}

String MIMETypeRegistry::mimeTypeForExtension(StringView extension)
{
    for (auto& entry : platformMediaTypes()) {
        if (equalIgnoringASCIICase(extension, entry.extension))
            return entry.type;
    }
    return emptyString();
}

bool MIMETypeRegistry::isApplicationPluginMIMEType(const String&)
{
    return false;
}

String MIMETypeRegistry::preferredExtensionForMIMEType(const String& mimeType)
{
    for (auto& entry : platformMediaTypes()) {
        if (equalIgnoringASCIICase(mimeType, entry.type))
            return entry.extension;
    }
    return emptyString();
}

Vector<String> MIMETypeRegistry::extensionsForMIMEType(const String&)
{
    ASSERT_NOT_IMPLEMENTED_YET();
    return { };
}

} // namespace WebCore

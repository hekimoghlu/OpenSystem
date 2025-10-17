/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 21, 2023.
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

#include <wtf/Assertions.h>
#include <wtf/HashMap.h>
#include <wtf/MainThread.h>
#include <wtf/WindowsExtras.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

static String mimeTypeForExtensionFromRegistry(const String& extension)
{
    auto ext = makeString('.', extension);
    WCHAR contentTypeStr[256];
    DWORD contentTypeStrLen = sizeof(contentTypeStr);
    DWORD keyType;

    HRESULT result = getRegistryValue(HKEY_CLASSES_ROOT, ext.wideCharacters().data(), L"Content Type", &keyType, contentTypeStr, &contentTypeStrLen);

    if (result == ERROR_SUCCESS && keyType == REG_SZ)
        return String(contentTypeStr, contentTypeStrLen / sizeof(contentTypeStr[0]) - 1);

    return String();
}

String MIMETypeRegistry::preferredExtensionForMIMEType(const String& type)
{
    auto path = makeString("MIME\\Database\\Content Type\\"_s, type);
    WCHAR extStr[MAX_PATH];
    DWORD extStrLen = sizeof(extStr);
    DWORD keyType;

    HRESULT result = getRegistryValue(HKEY_CLASSES_ROOT, path.wideCharacters().data(), L"Extension", &keyType, extStr, &extStrLen);

    if (result == ERROR_SUCCESS && keyType == REG_SZ)
        return String(extStr + 1, extStrLen / sizeof(extStr[0]) - 2);

    return String();
}

String MIMETypeRegistry::mimeTypeForExtension(StringView string)
{
    ASSERT(isMainThread());

    if (string.isEmpty())
        return String();

    auto ext = string.toString();
    static UncheckedKeyHashMap<String, String> mimetypeMap;
    if (mimetypeMap.isEmpty()) {
        //fill with initial values
        mimetypeMap.add("txt"_s, "text/plain"_s);
        mimetypeMap.add("pdf"_s, "application/pdf"_s);
        mimetypeMap.add("ps"_s, "application/postscript"_s);
        mimetypeMap.add("css"_s, "text/css"_s);
        mimetypeMap.add("html"_s, "text/html"_s);
        mimetypeMap.add("htm"_s, "text/html"_s);
        mimetypeMap.add("xml"_s, "text/xml"_s);
        mimetypeMap.add("xsl"_s, "text/xsl"_s);
        mimetypeMap.add("js"_s, "application/x-javascript"_s);
        mimetypeMap.add("xht"_s, "application/xhtml+xml"_s);
        mimetypeMap.add("xhtml"_s, "application/xhtml+xml"_s);
        mimetypeMap.add("rss"_s, "application/rss+xml"_s);
        mimetypeMap.add("webarchive"_s, "application/x-webarchive"_s);
#if USE(AVIF)
        mimetypeMap.add("avif"_s, "image/avif"_s);
#endif
        mimetypeMap.add("svg"_s, "image/svg+xml"_s);
        mimetypeMap.add("svgz"_s, "image/svg+xml"_s);
        mimetypeMap.add("jpg"_s, "image/jpeg"_s);
        mimetypeMap.add("jpeg"_s, "image/jpeg"_s);
        mimetypeMap.add("png"_s, "image/png"_s);
        mimetypeMap.add("tif"_s, "image/tiff"_s);
        mimetypeMap.add("tiff"_s, "image/tiff"_s);
        mimetypeMap.add("ico"_s, "image/ico"_s);
        mimetypeMap.add("cur"_s, "image/ico"_s);
        mimetypeMap.add("bmp"_s, "image/bmp"_s);
        mimetypeMap.add("wml"_s, "text/vnd.wap.wml"_s);
        mimetypeMap.add("wmlc"_s, "application/vnd.wap.wmlc"_s);
        mimetypeMap.add("m4a"_s, "audio/x-m4a"_s);
#if USE(JPEGXL)
        mimetypeMap.add("jxl"_s, "image/jxl"_s);
#endif
    }
    String result = mimetypeMap.get(ext);
    if (result.isEmpty()) {
        result = mimeTypeForExtensionFromRegistry(ext);
        if (!result.isEmpty())
            mimetypeMap.add(ext, result);
    }
    return result;
}

bool MIMETypeRegistry::isApplicationPluginMIMEType(const String&)
{
    return false;
}

Vector<String> MIMETypeRegistry::extensionsForMIMEType(const String&)
{
    ASSERT_NOT_IMPLEMENTED_YET();
    return { };
}

}

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 12, 2023.
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
#pragma once

#include <wtf/HashSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

struct TypeExtensionPair {
    ASCIILiteral type;
    ASCIILiteral extension;
};

WEBCORE_EXPORT const std::initializer_list<TypeExtensionPair>& commonMediaTypes();

struct MIMETypeRegistryThreadGlobalData {
    WTF_MAKE_TZONE_ALLOCATED(MIMETypeRegistryThreadGlobalData);
    WTF_MAKE_NONCOPYABLE(MIMETypeRegistryThreadGlobalData);
public:
    MIMETypeRegistryThreadGlobalData(HashSet<String, ASCIICaseInsensitiveHash>&& supportedImageMIMETypesForEncoding)
        : m_supportedImageMIMETypesForEncoding(WTFMove(supportedImageMIMETypesForEncoding))
    { }

    const HashSet<String, ASCIICaseInsensitiveHash>& supportedImageMIMETypesForEncoding() const { return m_supportedImageMIMETypesForEncoding; }

private:
    HashSet<String, ASCIICaseInsensitiveHash> m_supportedImageMIMETypesForEncoding;
};

class MIMETypeRegistry {
public:
    WEBCORE_EXPORT static String mimeTypeForExtension(StringView);
    WEBCORE_EXPORT static Vector<String> extensionsForMIMEType(const String& type);
    WEBCORE_EXPORT static String preferredExtensionForMIMEType(const String& type);
    WEBCORE_EXPORT static String mediaMIMETypeForExtension(StringView extension);

    WEBCORE_EXPORT static String mimeTypeForPath(StringView);

    static std::unique_ptr<MIMETypeRegistryThreadGlobalData> createMIMETypeRegistryThreadGlobalData();

    // Check to see if a MIME type is suitable for being loaded inline as an
    // image (e.g., <img> tags).
    WEBCORE_EXPORT static bool isSupportedImageMIMEType(const String& mimeType);

    // Check to see if a MIME type is suitable for being loaded as an image, including SVG and Video (where supported).
    WEBCORE_EXPORT static bool isSupportedImageVideoOrSVGMIMEType(const String& mimeType);

    // Check to see if a MIME type is suitable for being encoded.
    WEBCORE_EXPORT static bool isSupportedImageMIMETypeForEncoding(const String& mimeType);

    // Check to see if a MIME type is suitable for being loaded as a JavaScript or JSON resource.
    WEBCORE_EXPORT static bool isSupportedJavaScriptMIMEType(const String& mimeType);
    WEBCORE_EXPORT static bool isSupportedJSONMIMEType(const String& mimeType);

    // Check to see if a MIME type is suitable for being loaded as a WebAssembly module.
    WEBCORE_EXPORT static bool isSupportedWebAssemblyMIMEType(const String& mimeType);

    // Check to see if a MIME type is suitable for being loaded as a style sheet.
    static bool isSupportedStyleSheetMIMEType(const String& mimeType);

    // Check to see if a MIME type is suitable for being loaded as a font.
    static bool isSupportedFontMIMEType(const String& mimeType);

    // Check to see if a MIME type is a text media playlist type, such as an m3u8.
    static bool isTextMediaPlaylistMIMEType(const String& mimeType);

    // Check to see if a non-image MIME type is suitable for being loaded as a
    // document in a frame. Does not include supported JavaScript and JSON MIME types.
    WEBCORE_EXPORT static bool isSupportedNonImageMIMEType(const String& mimeType);

    // Check to see if a MIME type is suitable for being loaded using <video> and <audio>.
    WEBCORE_EXPORT static bool isSupportedMediaMIMEType(const String& mimeType);

    // Check to see if a MIME type is suitable for being loaded using <track>>.
    WEBCORE_EXPORT static bool isSupportedTextTrackMIMEType(const String& mimeType);

    // Check to see if a MIME type is a plugin implemented by the browser.
    static bool isApplicationPluginMIMEType(const String& mimeType);

    // Check to see if a MIME type is one of the common PDF/PS types.
    WEBCORE_EXPORT static bool isPDFMIMEType(const String& mimeType);

    WEBCORE_EXPORT static bool isUSDMIMEType(const String& mimeType);

    WEBCORE_EXPORT static bool isSupportedModelMIMEType(const String& mimeType);

    // Check to see if a MIME type is suitable for being shown inside a page.
    // Returns true if any of isSupportedImageMIMEType(), isSupportedNonImageMIMEType(),
    // isSupportedMediaMIMEType(), isSupportedJavaScriptMIMEType(), isSupportedJSONMIMEType(),
    // returns true or if the given MIME type begins with "text/" and
    // isUnsupportedTextMIMEType() returns false.
    WEBCORE_EXPORT static bool canShowMIMEType(const String& mimeType);

    // Check to see if a MIME type is one where an XML document should be created
    // rather than an HTML document.
    WEBCORE_EXPORT static bool isXMLMIMEType(const String& mimeType);

    // Check to see if a MIME type is for an XML external entity resource.
    WEBCORE_EXPORT static bool isXMLEntityMIMEType(StringView mimeType);

    // Used in page load algorithm to decide whether to display as a text
    // document in a frame. Not a good idea to use elsewhere, because that code
    // makes this test is after many other tests are done on the MIME type.
    WEBCORE_EXPORT static bool isTextMIMEType(const String& mimeType);

    WEBCORE_EXPORT static FixedVector<ASCIILiteral> supportedImageMIMETypes();
    static HashSet<String, ASCIICaseInsensitiveHash>& additionalSupportedImageMIMETypes();
    WEBCORE_EXPORT static HashSet<String, ASCIICaseInsensitiveHash>& supportedNonImageMIMETypes();
    WEBCORE_EXPORT static const HashSet<String>& supportedMediaMIMETypes();
    WEBCORE_EXPORT static FixedVector<ASCIILiteral> pdfMIMETypes();
    WEBCORE_EXPORT static FixedVector<ASCIILiteral> unsupportedTextMIMETypes();
    WEBCORE_EXPORT static FixedVector<ASCIILiteral> usdMIMETypes();

    WEBCORE_EXPORT static String appendFileExtensionIfNecessary(const String& filename, const String& mimeType);

    WEBCORE_EXPORT static String preferredImageMIMETypeForEncoding(const Vector<String>& mimeTypes, const Vector<String>& extensions);
    WEBCORE_EXPORT static bool containsImageMIMETypeForEncoding(const Vector<String>& mimeTypes, const Vector<String>& extensions);
    WEBCORE_EXPORT static Vector<String> allowedMIMETypes(const Vector<String>& mimeTypes, const Vector<String>& extensions);
    WEBCORE_EXPORT static Vector<String> allowedFileExtensions(const Vector<String>& mimeTypes, const Vector<String>& extensions);
    WEBCORE_EXPORT static bool isJPEGMIMEType(const String& mimeType);
    WEBCORE_EXPORT static bool isWebArchiveMIMEType(const String& mimeType);
private:
    // Check to see if the MIME type is not suitable for being loaded as a text
    // document in a frame. Only valid for MIME types begining with "text/".
    static bool isUnsupportedTextMIMEType(const String& mimeType);
};

WEBCORE_EXPORT const String& defaultMIMEType();

} // namespace WebCore

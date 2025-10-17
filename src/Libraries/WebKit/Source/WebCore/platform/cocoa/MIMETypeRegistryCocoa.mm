/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 15, 2023.
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
#import "config.h"
#import "MIMETypeRegistry.h"

#import <pal/spi/cocoa/CoreServicesSPI.h>
#import <pal/spi/cocoa/NSURLFileTypeMappingsSPI.h>
#import <wtf/RobinHoodHashMap.h>
#import <wtf/RobinHoodHashSet.h>
#import <wtf/cocoa/VectorCocoa.h>
#import <wtf/text/MakeString.h>

namespace WebCore {

static MemoryCompactLookupOnlyRobinHoodHashMap<String, MemoryCompactLookupOnlyRobinHoodHashSet<String>>& extensionsForMIMETypeMap()
{
    static NeverDestroyed extensionsForMIMETypeMap = [] {
        MemoryCompactLookupOnlyRobinHoodHashMap<String, MemoryCompactLookupOnlyRobinHoodHashSet<String>> map;

        auto addExtension = [&](const String& type, const String& extension) {
            map.add(type, MemoryCompactLookupOnlyRobinHoodHashSet<String>()).iterator->value.add(extension);
        };

        auto addExtensions = [&](const String& type, NSArray<NSString *> *extensions) {
            size_t pos = type.reverseFind('/');

            ASSERT(pos != notFound);
            auto wildcardMIMEType = makeString(StringView(type).left(pos), "/*"_s);

            for (NSString *extension in extensions) {
                // Add extension to wildcardMIMEType, for example add "png" to "image/*"
                addExtension(wildcardMIMEType, extension);
                // Add extension to its mimeType, for example add "png" to "image/png"
                addExtension(type, extension);
            }
        };

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
        auto allUTIs = adoptCF(_UTCopyDeclaredTypeIdentifiers());

        for (NSString *uti in (__bridge NSArray<NSString *> *)allUTIs.get()) {
            auto type = adoptCF(UTTypeCopyPreferredTagWithClass((__bridge CFStringRef)uti, kUTTagClassMIMEType));
            if (!type)
                continue;
            auto extensions = adoptCF(UTTypeCopyAllTagsWithClass((__bridge CFStringRef)uti, kUTTagClassFilenameExtension));
            if (!extensions || !CFArrayGetCount(extensions.get()))
                continue;
            addExtensions(type.get(), (__bridge NSArray<NSString *> *)extensions.get());
        }
ALLOW_DEPRECATED_DECLARATIONS_END

        return map;
    }();

    return extensionsForMIMETypeMap;
}

// Specify MIME type <-> extension mappings for type identifiers recognized by the system that are missing MIME type values.
static const UncheckedKeyHashMap<String, String, ASCIICaseInsensitiveHash>& additionalMimeTypesMap()
{
    static NeverDestroyed<UncheckedKeyHashMap<String, String, ASCIICaseInsensitiveHash>> mimeTypesMap = [] {
        UncheckedKeyHashMap<String, String, ASCIICaseInsensitiveHash> map;
        static constexpr TypeExtensionPair additionalTypes[] = {
            // FIXME: Remove this list once rdar://112044000 (Many camera RAW image type identifiers are missing MIME types) is resolved.
            { "image/x-canon-cr2"_s, "cr2"_s },
            { "image/x-canon-cr3"_s, "cr3"_s },
            { "image/x-epson-erf"_s, "erf"_s },
            { "image/x-fuji-raf"_s, "raf"_s },
            { "image/x-hasselblad-3fr"_s, "3fr"_s },
            { "image/x-hasselblad-fff"_s, "fff"_s },
            { "image/x-leaf-mos"_s, "mos"_s },
            { "image/x-leica-rwl"_s, "rwl"_s },
            { "image/x-minolta-mrw"_s, "mrw"_s },
            { "image/x-nikon-nef"_s, "nef"_s },
            { "image/x-olympus-orf"_s, "orf"_s },
            { "image/x-panasonic-raw"_s, "raw"_s },
            { "image/x-panasonic-rw2"_s, "rw2"_s },
            { "image/x-pentax-pef"_s, "pef"_s },
            { "image/x-phaseone-iiq"_s, "iiq"_s },
            { "image/x-samsung-srw"_s, "srw"_s },
            { "image/x-sony-arw"_s, "arw"_s },
            { "image/x-sony-srf"_s, "srf"_s },
        };
        for (auto& [type, extension] : additionalTypes)
            map.add(extension, type);
        return map;
    }();
    return mimeTypesMap;
}

static const UncheckedKeyHashMap<String, Vector<String>, ASCIICaseInsensitiveHash>& additionalExtensionsMap()
{
    static NeverDestroyed<UncheckedKeyHashMap<String, Vector<String>, ASCIICaseInsensitiveHash>> extensionsMap = [] {
        UncheckedKeyHashMap<String, Vector<String>, ASCIICaseInsensitiveHash> map;
        for (auto& [extension, type] : additionalMimeTypesMap()) {
            map.ensure(type, [] {
                return Vector<String>();
            }).iterator->value.append(extension);
        }
        return map;
    }();
    return extensionsMap;
}

static Vector<String> extensionsForWildcardMIMEType(const String& type)
{
    Vector<String> extensions;

    auto iterator = extensionsForMIMETypeMap().find(type);
    if (iterator != extensionsForMIMETypeMap().end())
        extensions.appendRange(iterator->value.begin(), iterator->value.end());

    return extensions;
}

String MIMETypeRegistry::mimeTypeForExtension(StringView extension)
{
    auto string = extension.createNSStringWithoutCopying();

    NSString *mimeType = [[NSURLFileTypeMappings sharedMappings] MIMETypeForExtension:string.get()];
    if (mimeType.length)
        return mimeType;

    auto mapEntry = additionalMimeTypesMap().find<ASCIICaseInsensitiveStringViewHashTranslator>(extension);
    if (mapEntry != additionalMimeTypesMap().end())
        return mapEntry->value;

    return nullString();
}

Vector<String> MIMETypeRegistry::extensionsForMIMEType(const String& type)
{
    if (type.isNull())
        return { };

    if (type.endsWith('*'))
        return extensionsForWildcardMIMEType(type);

    NSArray *extensions = [[NSURLFileTypeMappings sharedMappings] extensionsForMIMEType:type];
    if (extensions.count)
        return makeVector<String>(extensions);

    auto mapEntry = additionalExtensionsMap().find(type);
    if (mapEntry != additionalExtensionsMap().end())
        return mapEntry->value;

    return { };
}

String MIMETypeRegistry::preferredExtensionForMIMEType(const String& type)
{
    if (type.isNull())
        return nullString();

    // We accept some non-standard USD MIMETypes, so we can't rely on
    // the file type mappings.
    if (isUSDMIMEType(type))
        return "usdz"_s;

    NSString *preferredExtension = [[NSURLFileTypeMappings sharedMappings] preferredExtensionForMIMEType:(NSString *)type];
    if (preferredExtension.length)
        return preferredExtension;

    auto mapEntry = additionalExtensionsMap().find(type);
    if (mapEntry != additionalExtensionsMap().end())
        return mapEntry->value.first();

    return nullString();
}

bool MIMETypeRegistry::isApplicationPluginMIMEType(const String& MIMEType)
{
#if ENABLE(PDF_PLUGIN)
    // FIXME: This should test if we're actually going to use PDFPlugin,
    // but we only know that in WebKit2 at the moment. This is not a problem
    // in practice because if we don't have PDFPlugin and we go to instantiate the
    // plugin, there won't exist an application plugin supporting these MIME types.
    if (isPDFMIMEType(MIMEType))
        return true;
#else
    UNUSED_PARAM(MIMEType);
#endif

    return false;
}

}

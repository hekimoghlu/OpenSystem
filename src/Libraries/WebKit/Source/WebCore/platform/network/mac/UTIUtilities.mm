/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 27, 2024.
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
#import "UTIUtilities.h"

#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#import <wtf/HashSet.h>
#import <wtf/Lock.h>
#import <wtf/MainThread.h>
#import <wtf/SortedArrayMap.h>
#import <wtf/TinyLRUCache.h>
#import <wtf/cf/TypeCastsCF.h>
#import <wtf/text/WTFString.h>
#include <wtf/cocoa/VectorCocoa.h>

#if PLATFORM(IOS_FAMILY)
#import <MobileCoreServices/MobileCoreServices.h>
#endif

#if HAVE(CGIMAGESOURCE_WITH_SET_ALLOWABLE_TYPES)
#include <pal/spi/cg/ImageIOSPI.h>
#endif

namespace WebCore {

String MIMETypeFromUTI(const String& uti)
{
    RetainPtr type = [UTType typeWithIdentifier:uti];
    return type.get().preferredMIMEType;
}

UncheckedKeyHashSet<String> RequiredMIMETypesFromUTI(const String& uti)
{
    UncheckedKeyHashSet<String> mimeTypes;

    auto mainMIMEType = MIMETypeFromUTI(uti);
    if (!mainMIMEType.isEmpty())
        mimeTypes.add(mainMIMEType);

    if (equalLettersIgnoringASCIICase(uti, "com.adobe.photoshop-image"_s))
        mimeTypes.add("application/x-photoshop"_s);

    return mimeTypes;
}

RetainPtr<CFStringRef> mimeTypeFromUTITree(CFStringRef uti)
{
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    // Check if this UTI has a MIME type.
    if (auto type = adoptCF(UTTypeCopyPreferredTagWithClass(uti, kUTTagClassMIMEType)))
        return type;

    // If not, walk the ancestory of this UTI via its "ConformsTo" tags and return the first MIME type we find.
    auto declaration = adoptCF(UTTypeCopyDeclaration(uti));
    if (!declaration)
        return nullptr;

    auto value = CFDictionaryGetValue(declaration.get(), kUTTypeConformsToKey);
ALLOW_DEPRECATED_DECLARATIONS_END
    if (!value)
        return nullptr;

    if (auto string = dynamic_cf_cast<CFStringRef>(value))
        return mimeTypeFromUTITree(string);

    if (auto array = dynamic_cf_cast<CFArrayRef>(value)) {
        CFIndex count = CFArrayGetCount(array);
        for (CFIndex i = 0; i < count; ++i) {
            if (auto string = dynamic_cf_cast<CFStringRef>(CFArrayGetValueAtIndex(array, i))) {
                if (auto type = mimeTypeFromUTITree(string))
                    return type;
            }
        }
    }

    return nullptr;
}

static NSString *UTIFromPotentiallyUnknownMIMEType(StringView mimeType)
{
    static constexpr std::pair<ComparableLettersLiteral, NSString *> typesArray[] = {
        { "model/usd"_s, @"com.pixar.universal-scene-description-mobile" },
        { "model/vnd.pixar.usd"_s, @"com.pixar.universal-scene-description-mobile" },
        { "model/vnd.reality"_s, @"com.apple.reality" },
        { "model/vnd.usdz+zip"_s, @"com.pixar.universal-scene-description-mobile" },
    };
    static constexpr SortedArrayMap typesMap { typesArray };
    return typesMap.get(mimeType, nil);
}

struct UTIFromMIMETypeCachePolicy : TinyLRUCachePolicy<String, RetainPtr<NSString>> {
public:
    static RetainPtr<NSString> createValueForKey(const String& mimeType)
    {
        if (auto type = UTIFromPotentiallyUnknownMIMEType(mimeType))
            return type;

        if (RetainPtr type = [UTType typeWithMIMEType:mimeType])
            return type.get().identifier;

        return @"";
    }

    static String createKeyForStorage(const String& key) { return key.isolatedCopy(); }
};

static Lock cacheUTIFromMIMETypeLock;
static TinyLRUCache<String, RetainPtr<NSString>, 16, UTIFromMIMETypeCachePolicy>& cacheUTIFromMIMEType() WTF_REQUIRES_LOCK(cacheUTIFromMIMETypeLock)
{
    static NeverDestroyed<TinyLRUCache<String, RetainPtr<NSString>, 16, UTIFromMIMETypeCachePolicy>> cache;
    return cache;
}

String UTIFromMIMEType(const String& mimeType)
{
    Locker locker { cacheUTIFromMIMETypeLock };
    return cacheUTIFromMIMEType().get(mimeType).get();
}

bool isDeclaredUTI(const String& UTI)
{
    RetainPtr type = [UTType typeWithIdentifier:UTI];
    return type.get().isDeclared;
}

void setImageSourceAllowableTypes(const Vector<String>& supportedImageTypes)
{
#if HAVE(CGIMAGESOURCE_WITH_SET_ALLOWABLE_TYPES)
    // A WebPage might be reinitialized. So restrict ImageIO to the default and
    // the additional supported image formats only once.
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [supportedImageTypes] {
        auto allowableTypes = createNSArray(supportedImageTypes);
        auto status = CGImageSourceSetAllowableTypes((__bridge CFArrayRef)allowableTypes.get());
        RELEASE_ASSERT_WITH_MESSAGE(supportedImageTypes.isEmpty() || status == noErr, "CGImageSourceSetAllowableTypes() returned error: %d.", status);
    });
#else
    UNUSED_PARAM(supportedImageTypes);
#endif
}

} // namespace WebCore

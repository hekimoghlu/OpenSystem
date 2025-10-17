/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 13, 2024.
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
#import "AVStreamDataParserMIMETypeCache.h"

#if ENABLE(MEDIA_SOURCE) && USE(AVFOUNDATION)

#import "AVAssetMIMETypeCache.h"
#import "ContentType.h"
#import <pal/spi/cocoa/AVFoundationSPI.h>
#import <wtf/HashSet.h>

#import <pal/cf/CoreMediaSoftLink.h>
#import <pal/cocoa/AVFoundationSoftLink.h>

NS_ASSUME_NONNULL_BEGIN
@interface AVStreamDataParser (AVStreamDataParserExtendedMIMETypes)
+ (BOOL)canParseExtendedMIMEType:(NSString *)extendedMIMEType;
@end
NS_ASSUME_NONNULL_END

namespace WebCore {

AVStreamDataParserMIMETypeCache& AVStreamDataParserMIMETypeCache::singleton()
{
    static NeverDestroyed<AVStreamDataParserMIMETypeCache> cache;
    return cache.get();
}

bool AVStreamDataParserMIMETypeCache::isAvailable() const
{
#if ENABLE(VIDEO) && USE(AVFOUNDATION)
    if (!PAL::AVFoundationLibrary())
        return false;

    return [PAL::getAVStreamDataParserClass() respondsToSelector:@selector(audiovisualMIMETypes)];
#else
    return false;
#endif
}

MediaPlayerEnums::SupportsType AVStreamDataParserMIMETypeCache::canDecodeType(const String& type)
{
    if (isAvailable())
        return MIMETypeCache::canDecodeType(type);

    auto& assetCache = AVAssetMIMETypeCache::singleton();
    if (assetCache.isAvailable())
        return assetCache.canDecodeType(type);

    return MediaPlayerEnums::SupportsType::IsNotSupported;
}

HashSet<String>& AVStreamDataParserMIMETypeCache::supportedTypes()
{
    if (isAvailable())
        return MIMETypeCache::supportedTypes();

    auto& assetCache = AVAssetMIMETypeCache::singleton();
    if (assetCache.isAvailable())
        return assetCache.supportedTypes();

    return MIMETypeCache::supportedTypes();
}

bool AVStreamDataParserMIMETypeCache::canDecodeExtendedType(const ContentType& type)
{
#if ENABLE(VIDEO) && USE(AVFOUNDATION)
    ASSERT(isAvailable());

    if ([PAL::getAVStreamDataParserClass() respondsToSelector:@selector(canParseExtendedMIMEType:)])
        return [PAL::getAVStreamDataParserClass() canParseExtendedMIMEType:type.raw()];

    // FIXME(rdar://50502771) AVStreamDataParser does not have an -canParseExtendedMIMEType: method on this system,
    //  so just replace the container type with a valid one from AVAssetMIMETypeCache and ask that cache if it
    //  can decode this type.
    auto& assetCache = AVAssetMIMETypeCache::singleton();
    if (!assetCache.isAvailable() || assetCache.supportedTypes().isEmpty())
        return false;

    String replacementType = makeStringByReplacingAll(type.raw(), type.containerType(), *assetCache.supportedTypes().begin());
    return assetCache.canDecodeType(replacementType) == MediaPlayerEnums::SupportsType::IsSupported;
#endif

    return false;
}

void AVStreamDataParserMIMETypeCache::initializeCache(HashSet<String>& cache)
{
#if ENABLE(VIDEO) && USE(AVFOUNDATION)
    if (!isAvailable())
        return;

    for (NSString* type in [PAL::getAVStreamDataParserClass() audiovisualMIMETypes])
        cache.add(type);
#endif
}

}

#endif

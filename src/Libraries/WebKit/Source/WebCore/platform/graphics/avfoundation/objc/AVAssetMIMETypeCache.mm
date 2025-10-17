/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 3, 2023.
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
#import "AVAssetMIMETypeCache.h"

#if PLATFORM(COCOA)

#import "ContentType.h"
#import "SourceBufferParserWebM.h"
#import "WebMAudioUtilitiesCocoa.h"
#import <pal/spi/cocoa/AVFoundationSPI.h>
#import <pal/spi/cocoa/AudioToolboxSPI.h>
#import <wtf/SortedArrayMap.h>
#import <wtf/text/MakeString.h>

#import <pal/cf/AudioToolboxSoftLink.h>
#import <pal/cf/CoreMediaSoftLink.h>
#import <pal/cocoa/AVFoundationSoftLink.h>

namespace WebCore {

AVAssetMIMETypeCache& AVAssetMIMETypeCache::singleton()
{
    static NeverDestroyed<AVAssetMIMETypeCache> cache;
    return cache.get();
}

bool AVAssetMIMETypeCache::isAvailable() const
{
#if ENABLE(VIDEO) && USE(AVFOUNDATION)
    return PAL::isAVFoundationFrameworkAvailable();
#else
    return false;
#endif
}

#if ENABLE(VIDEO) && USE(AVFOUNDATION) && ENABLE(OPUS)
static bool isMultichannelOpusAvailable()
{
    static bool isMultichannelOpusAvailable = [] {
        if (!isOpusDecoderAvailable())
            return false;

        AudioStreamBasicDescription asbd { };
        asbd.mFormatID = kAudioFormatOpus;

        // AvailableDecodeChannelLayoutTags is an array of AudioChannelLayoutTag objects
        UInt32 propertySize = 0;
        auto error = PAL::AudioFormatGetPropertyInfo(kAudioFormatProperty_AvailableDecodeChannelLayoutTags, sizeof(asbd), &asbd, &propertySize);
        if (error != noErr || propertySize < sizeof(AudioChannelLayoutTag))
            return false;

        size_t count = propertySize / sizeof(AudioChannelLayoutTag);
        Vector<AudioChannelLayoutTag> channelLayoutTags(count, { });

        error = PAL::AudioFormatGetProperty(kAudioFormatProperty_AvailableDecodeChannelLayoutTags, sizeof(asbd), &asbd, &propertySize, channelLayoutTags.data());
        if (error != noErr)
            return false;

        size_t maximumDecodeChannelCount = 0;
        for (auto& channelLayoutTag : channelLayoutTags) {
            UInt32 layoutIndicator = (channelLayoutTag & 0xFFFF0000);
            if (layoutIndicator == kAudioChannelLayoutTag_Unknown || layoutIndicator == kAudioChannelLayoutTag_DiscreteInOrder)
                continue;
            maximumDecodeChannelCount = std::max<size_t>(maximumDecodeChannelCount, AudioChannelLayoutTag_GetNumberOfChannels(channelLayoutTag));
        }

        return maximumDecodeChannelCount > 2;
    }();
    return isMultichannelOpusAvailable;
}
#endif

bool AVAssetMIMETypeCache::canDecodeExtendedType(const ContentType& typeParameter)
{
    ContentType type = typeParameter;
#if ENABLE(VIDEO) && USE(AVFOUNDATION)
#if ENABLE(OPUS)
    // Disclaim support for 'opus' if multi-channel decode is not available.
    if ((type.containerType() == "video/mp4"_s || type.containerType() == "audio/mp4"_s)
        && type.codecs().contains("opus"_s) && !isMultichannelOpusAvailable())
        return false;
#endif

    // Some platforms will disclaim support for 'flac', and only support the MP4RA registered `fLaC`
    // codec string for flac, so convert the former to the latter before querying.
    if ((type.containerType() == "video/mp4"_s || type.containerType() == "audio/mp4"_s)
        && type.codecs().contains("flac"_s))
        type = ContentType(makeStringByReplacingAll(type.raw(), "flac"_s, "fLaC"_s));

    ASSERT(isAvailable());

#if HAVE(AVURLASSET_ISPLAYABLEEXTENDEDMIMETYPEWITHOPTIONS)
    if (PAL::canLoad_AVFoundation_AVURLAssetExtendedMIMETypePlayabilityTreatPlaylistMIMETypesAsISOBMFFMediaDataContainersKey()
        && [PAL::getAVURLAssetClass() respondsToSelector:@selector(isPlayableExtendedMIMEType:options:)]) {
        if ([PAL::getAVURLAssetClass() isPlayableExtendedMIMEType:type.raw() options:@{ AVURLAssetExtendedMIMETypePlayabilityTreatPlaylistMIMETypesAsISOBMFFMediaDataContainersKey: @YES }])
            return true;
    } else
#endif
    if ([PAL::getAVURLAssetClass() isPlayableExtendedMIMEType:type.raw()])
        return true;

#endif // ENABLE(VIDEO) && USE(AVFOUNDATION)

    return false;
}

bool AVAssetMIMETypeCache::isUnsupportedContainerType(const String& type)
{
    if (type.isEmpty())
        return false;

    String lowerCaseType = type.convertToASCIILowercase();

    // AVFoundation will return non-video MIME types which it claims to support, but which we
    // do not support in the <video> element. Reject all non video/, audio/, and application/ types.
    if (!lowerCaseType.startsWith("video/"_s) && !lowerCaseType.startsWith("audio/"_s) && !lowerCaseType.startsWith("application/"_s))
        return true;

    // Reject types we know AVFoundation does not support that sites commonly ask about.
    static constexpr ComparableASCIILiteral unsupportedTypesArray[] = { "video/h264"_s, "video/x-flv"_s };
    static constexpr SortedArraySet unsupportedTypesSet { unsupportedTypesArray };
    return unsupportedTypesSet.contains(lowerCaseType);
}

bool AVAssetMIMETypeCache::isStaticContainerType(StringView type)
{
    static constexpr ComparableLettersLiteral staticContainerTypesArray[] = {
        "application/vnd.apple.mpegurl"_s,
        "application/x-mpegurl"_s,
        "audio/3gpp"_s,
        "audio/aac"_s,
        "audio/aacp"_s,
        "audio/aiff"_s,
        "audio/basic"_s,
        "audio/mp3"_s,
        "audio/mp4"_s,
        "audio/mpeg"_s,
        "audio/mpeg3"_s,
        "audio/mpegurl"_s,
        "audio/mpg"_s,
        "audio/vnd.wave"_s,
        "audio/wav"_s,
        "audio/wave"_s,
        "audio/x-aac"_s,
        "audio/x-aiff"_s,
        "audio/x-m4a"_s,
        "audio/x-mpegurl"_s,
        "audio/x-wav"_s,
        "video/3gpp"_s,
        "video/3gpp2"_s,
        "video/mp4"_s,
        "video/mpeg"_s,
        "video/mpeg2"_s,
        "video/mpg"_s,
        "video/quicktime"_s,
        "video/x-m4v"_s,
        "video/x-mpeg"_s,
        "video/x-mpg"_s,
    };
    static constexpr SortedArraySet staticContainerTypesSet { staticContainerTypesArray };
    return staticContainerTypesSet.contains(type);
}

void AVAssetMIMETypeCache::addSupportedTypes(const Vector<String>& types)
{
    MIMETypeCache::addSupportedTypes(types);
    if (m_cacheTypeCallback)
        m_cacheTypeCallback(types);
}

void AVAssetMIMETypeCache::initializeCache(HashSet<String>& cache)
{
#if ENABLE(VIDEO) && USE(AVFOUNDATION)
    if (!isAvailable())
        return;

    for (NSString *type in [PAL::getAVURLAssetClass() audiovisualMIMETypes])
        cache.add(type);

    if (m_cacheTypeCallback)
        m_cacheTypeCallback(copyToVector(cache));
#endif
}

}

#endif

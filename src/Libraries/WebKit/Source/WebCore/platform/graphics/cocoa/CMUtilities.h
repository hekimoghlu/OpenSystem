/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 30, 2021.
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

#if PLATFORM(COCOA)

#include "CAAudioStreamDescription.h"
#include "MediaSample.h"
#include <CoreAudio/CoreAudioTypes.h>
#include <memory>
#include <wtf/Expected.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

typedef struct AudioFormatVorbisModeInfo AudioFormatVorbisModeInfo;
typedef const struct opaqueCMFormatDescription* CMFormatDescriptionRef;
typedef struct opaqueCMSampleBuffer* CMSampleBufferRef;
typedef struct __CVBuffer* CVPixelBufferRef;
typedef struct OpaqueCMBlockBuffer* CMBlockBufferRef;

namespace WebCore {

class MediaSamplesBlock;
class SharedBuffer;
struct AudioInfo;
struct PlatformVideoColorSpace;
struct TrackInfo;

WEBCORE_EXPORT RetainPtr<CMFormatDescriptionRef> createFormatDescriptionFromTrackInfo(const TrackInfo&);
WEBCORE_EXPORT RefPtr<AudioInfo> createAudioInfoFromFormatDescription(CMFormatDescriptionRef);
// audioStreamDescriptFromAudioInfo only works with compressed audio format (non PCM)
WEBCORE_EXPORT CAAudioStreamDescription audioStreamDescriptionFromAudioInfo(const AudioInfo&);
WEBCORE_EXPORT Ref<SharedBuffer> sharedBufferFromCMBlockBuffer(CMBlockBufferRef);
WEBCORE_EXPORT RetainPtr<CMBlockBufferRef> ensureContiguousBlockBuffer(CMBlockBufferRef);

// Convert MediaSamplesBlock to the equivalent CMSampleBufferRef. If CMFormatDescriptionRef
// is set it will be used, otherwise it will be created from the MediaSamplesBlock's TrackInfo.
WEBCORE_EXPORT Expected<RetainPtr<CMSampleBufferRef>, CString> toCMSampleBuffer(const MediaSamplesBlock&, CMFormatDescriptionRef = nullptr);
// Convert CMSampleBufferRef to the equivalent MediaSamplesBlock. If TrackInfo
// is set it will be used, otherwise it will be created from the CMSampleBufferRef's CMFormatDescriptionRef.
WEBCORE_EXPORT UniqueRef<MediaSamplesBlock> samplesBlockFromCMSampleBuffer(CMSampleBufferRef, TrackInfo* = nullptr);

WEBCORE_EXPORT void attachColorSpaceToPixelBuffer(const PlatformVideoColorSpace&, CVPixelBufferRef);

class PacketDurationParser final {
    WTF_MAKE_TZONE_ALLOCATED(PacketDurationParser);
public:
    explicit PacketDurationParser(const AudioInfo&);
    ~PacketDurationParser();

    bool isValid() const { return m_isValid; }
    size_t framesInPacket(std::span<const uint8_t>);
    void reset();

private:
    uint32_t m_audioFormatID { 0 };
    uint32_t m_constantFramesPerPacket { 0 };
    std::optional<Seconds> m_frameDuration;
    uint32_t m_sampleRate { 0 };
#if ENABLE(VORBIS)
#if HAVE(AUDIOFORMATPROPERTY_VARIABLEPACKET_SUPPORTED)
    std::unique_ptr<AudioFormatVorbisModeInfo> m_vorbisModeInfo;
    uint32_t m_vorbisModeMask { 0 };
#endif
    uint32_t m_lastVorbisBlockSize { 0 };
#endif
    bool m_isValid { false };
};

Vector<AudioStreamPacketDescription> getPacketDescriptions(CMSampleBufferRef);

}

#endif

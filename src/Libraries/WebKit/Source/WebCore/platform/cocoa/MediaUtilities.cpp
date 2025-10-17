/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 3, 2025.
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
#include "MediaUtilities.h"

#include "AudioStreamDescription.h"
#include "WebAudioBufferList.h"
#include <wtf/SoftLinking.h>
#include <pal/cf/CoreMediaSoftLink.h>

namespace WebCore {

RetainPtr<CMFormatDescriptionRef> createAudioFormatDescription(const AudioStreamDescription& description, std::span<const uint8_t> magicCookie)
{
    auto basicDescription = std::get<const AudioStreamBasicDescription*>(description.platformDescription().description);
    CMFormatDescriptionRef format = nullptr;
    auto error = PAL::CMAudioFormatDescriptionCreate(kCFAllocatorDefault, basicDescription, 0, nullptr, magicCookie.size(), magicCookie.data(), nullptr, &format);
    if (error) {
        LOG_ERROR("createAudioFormatDescription failed with %d", static_cast<int>(error));
        return nullptr;
    }
    return adoptCF(format);
}

RetainPtr<CMSampleBufferRef> createAudioSampleBuffer(const PlatformAudioData& data, const AudioStreamDescription& description, CMTime time, size_t sampleCount)
{
    // FIXME: check if we can reuse the format for multiple sample buffers.
    auto format = createAudioFormatDescription(description);
    if (!format)
        return nullptr;

    CMSampleBufferRef sampleBuffer = nullptr;
    auto error = PAL::CMAudioSampleBufferCreateWithPacketDescriptions(kCFAllocatorDefault, nullptr, false, nullptr, nullptr, format.get(), sampleCount, time, nullptr, &sampleBuffer);
    if (error) {
        LOG_ERROR("createAudioSampleBuffer with packet descriptions failed - %d", static_cast<int>(error));
        return nullptr;
    }
    auto buffer = adoptCF(sampleBuffer);

    error = PAL::CMSampleBufferSetDataBufferFromAudioBufferList(buffer.get(), kCFAllocatorDefault, kCFAllocatorDefault, 0, downcast<WebAudioBufferList>(data).list());
    if (error) {
        LOG_ERROR("createAudioSampleBuffer from audio buffer list failed - %d", static_cast<int>(error));
        return nullptr;
    }
    return buffer;
}

RetainPtr<CMSampleBufferRef> createVideoSampleBuffer(CVPixelBufferRef pixelBuffer, CMTime presentationTime)
{
    CMVideoFormatDescriptionRef formatDescription = nullptr;
    auto status = PAL::CMVideoFormatDescriptionCreateForImageBuffer(kCFAllocatorDefault, pixelBuffer, &formatDescription);
    if (status)
        return nullptr;
    auto retainedFormatDescription = adoptCF(formatDescription);

    CMSampleTimingInfo timingInfo { PAL::kCMTimeInvalid, presentationTime, PAL::kCMTimeInvalid };
    CMSampleBufferRef sampleBuffer;
    status = PAL::CMSampleBufferCreateReadyWithImageBuffer(kCFAllocatorDefault, pixelBuffer, formatDescription, &timingInfo, &sampleBuffer);
    if (status)
        return nullptr;

    return adoptCF(sampleBuffer);
}

} // namespace WebCore

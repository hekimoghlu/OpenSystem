/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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

#if ENABLE(GPU_PROCESS)

#include <WebCore/MediaSample.h>

namespace WebKit {

struct RemoteAudioInfo {
    RemoteAudioInfo() = default;
    explicit RemoteAudioInfo(const WebCore::AudioInfo& info)
        : codecName(info.codecName)
        , codecString(info.codecString)
        , trackID(info.trackID)
        , rate(info.rate)
        , channels(info.channels)
        , framesPerPacket(info.framesPerPacket)
        , bitDepth(info.bitDepth)
        , cookieData(info.cookieData)
    {
    }
    // Used by IPC generator
    RemoteAudioInfo(WebCore::FourCC codecName, const String& codecString, WebCore::TrackID trackID, uint32_t rate, uint32_t channels, uint32_t framesPerPacket, uint8_t bitDepth, RefPtr<WebCore::SharedBuffer> cookieData)
        : codecName(codecName)
        , codecString(codecString)
        , trackID(trackID)
        , rate(rate)
        , channels(channels)
        , framesPerPacket(framesPerPacket)
        , bitDepth(bitDepth)
        , cookieData(WTFMove(cookieData))
    {
    }

    Ref<WebCore::AudioInfo> toAudioInfo() const
    {
        Ref audioInfo = WebCore::AudioInfo::create();
        audioInfo->codecName = codecName;
        audioInfo->codecString = codecString;
        audioInfo->trackID = trackID;
        audioInfo->rate = rate;
        audioInfo->framesPerPacket = framesPerPacket;
        audioInfo->bitDepth = bitDepth;
        audioInfo->cookieData = cookieData;
        return audioInfo;
    }

    WebCore::FourCC codecName;
    String codecString;
    WebCore::TrackID trackID { 0 };
    uint32_t rate { 0 };
    uint32_t channels { 0 };
    uint32_t framesPerPacket { 0 };
    uint8_t bitDepth { 16 };
    RefPtr<WebCore::SharedBuffer> cookieData;
};

struct RemoteVideoInfo {
    RemoteVideoInfo() = default;
    explicit RemoteVideoInfo(const WebCore::VideoInfo& info)
        : codecName(info.codecName)
        , codecString(info.codecString)
        , trackID(info.trackID)
        , size(info.size)
        , displaySize(info.displaySize)
        , bitDepth(info.bitDepth)
        , colorSpace(info.colorSpace)
        , atomData(info.atomData)
    {
    }
    // Used by IPC generator
    RemoteVideoInfo(WebCore::FourCC codecName, const String& codecString, WebCore::TrackID trackID, WebCore::FloatSize size, WebCore::FloatSize displaySize, uint8_t bitDepth, WebCore::PlatformVideoColorSpace colorSpace, RefPtr<WebCore::SharedBuffer> atomData)
        : codecName(codecName)
        , codecString(codecString)
        , trackID(trackID)
        , size(size)
        , displaySize(displaySize)
        , bitDepth(bitDepth)
        , colorSpace(colorSpace)
        , atomData(WTFMove(atomData))
    {
    }

    Ref<WebCore::VideoInfo> toVideoInfo() const
    {
        Ref videoInfo = WebCore::VideoInfo::create();
        videoInfo->codecName = codecName;
        videoInfo->codecString = codecString;
        videoInfo->trackID = trackID;
        videoInfo->size = size;
        videoInfo->displaySize = displaySize;
        videoInfo->bitDepth = bitDepth;
        videoInfo->colorSpace = colorSpace;
        videoInfo->atomData = atomData;
        return videoInfo;
    }

    WebCore::FourCC codecName;
    String codecString;
    WebCore::TrackID trackID { 0 };
    WebCore::FloatSize size;
    WebCore::FloatSize displaySize;
    uint8_t bitDepth { 8 };
    WebCore::PlatformVideoColorSpace colorSpace;
    RefPtr<WebCore::SharedBuffer> atomData;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)

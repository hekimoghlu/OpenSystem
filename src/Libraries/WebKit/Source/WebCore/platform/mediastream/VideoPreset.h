/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 22, 2025.
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

#if ENABLE(MEDIA_STREAM)

#include "ImageBuffer.h"
#include "RealtimeMediaSource.h"
#include <wtf/Lock.h>
#include <wtf/RetainPtr.h>
#include <wtf/RunLoop.h>

OBJC_CLASS AVCaptureDeviceFormat;

namespace WebCore {

struct FrameRateRange {
    double minimum;
    double maximum;
};

struct VideoPresetData {
    IntSize size;
    Vector<FrameRateRange> frameRateRanges;
    double minZoom { 1 };
    double maxZoom { 1 };
    bool isEfficient { false };
};

class VideoPreset {
public:
    explicit VideoPreset(VideoPresetData&& data)
        : m_data(WTFMove(data))
    {
    }
    VideoPreset(IntSize size, Vector<FrameRateRange>&& frameRateRanges, std::optional<double> minZoom, std::optional<double> maxZoom, bool isEfficient)
        : m_data { size, WTFMove(frameRateRanges), minZoom.value_or(1), maxZoom.value_or(1), isEfficient }
    {
        ASSERT(m_data.maxZoom >= m_data.minZoom);
    }

    IntSize size() const { return m_data.size; }
    const Vector<FrameRateRange>& frameRateRanges() const { return m_data.frameRateRanges; }
    double minZoom() const { return m_data.minZoom; }
    double maxZoom() const { return m_data.maxZoom; }

    void sortFrameRateRanges();

#if PLATFORM(COCOA) && USE(AVFOUNDATION)
    void setFormat(AVCaptureDeviceFormat* format) { m_format = format; }
    AVCaptureDeviceFormat* format() const { return m_format.get(); }
#endif

    double maxFrameRate() const;
    double minFrameRate() const;

    bool isZoomSupported() const { return m_data.minZoom != 1 || m_data.maxZoom != 1; }

    bool isEfficient() const { return m_data.isEfficient; }
    void log()const;

protected:
    VideoPresetData m_data;
#if PLATFORM(COCOA) && USE(AVFOUNDATION)
    RetainPtr<AVCaptureDeviceFormat> m_format;
#endif
};

inline void VideoPreset::log() const
{
    WTFLogAlways("VideoPreset of size (%d,%d), zoom is [%f, %f]", m_data.size.width(), m_data.size.height(), m_data.minZoom, m_data.maxZoom);
    for (auto range : m_data.frameRateRanges)
        WTFLogAlways("VideoPreset frame rate range [%f, %f]", range.minimum, range.maximum);
}

inline double VideoPreset::minFrameRate() const
{
    double minFrameRate = std::numeric_limits<double>::max();
    for (auto& range : m_data.frameRateRanges) {
        if (minFrameRate > range.minimum)
            minFrameRate = range.minimum;
    }
    return minFrameRate;
}

inline double VideoPreset::maxFrameRate() const
{
    double maxFrameRate = 0;
    for (auto& range : m_data.frameRateRanges) {
        if (maxFrameRate < range.maximum)
            maxFrameRate = range.maximum;
    }
    return maxFrameRate;
}

inline void VideoPreset::sortFrameRateRanges()
{
    std::sort(m_data.frameRateRanges.begin(), m_data.frameRateRanges.end(),
        [&] (const auto& a, const auto& b) -> bool {
            return a.minimum < b.minimum;
    });
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)


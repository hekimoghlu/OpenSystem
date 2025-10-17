/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 7, 2022.
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
#include "VideoPreset.h"
#include <wtf/Lock.h>
#include <wtf/RunLoop.h>

namespace WebCore {

class ImageTransferSessionVT;

enum class VideoFrameRotation : uint16_t;

class WEBCORE_EXPORT RealtimeVideoCaptureSource : public RealtimeMediaSource, public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<RealtimeVideoCaptureSource, WTF::DestructionThread::MainRunLoop> {
public:
    virtual ~RealtimeVideoCaptureSource();

    bool supportsSizeFrameRateAndZoom(const VideoPresetConstraints&) override;
    virtual void generatePresets() = 0;

    double observedFrameRate() const final { return m_observedFrameRate; }
    Vector<VideoPresetData> presetsData();

    void ensureIntrinsicSizeMaintainsAspectRatio();

    WTF_ABSTRACT_THREAD_SAFE_REF_COUNTED_AND_CAN_MAKE_WEAK_PTR_IMPL;

protected:
    RealtimeVideoCaptureSource(const CaptureDevice&, MediaDeviceHashSalts&&, std::optional<PageIdentifier>);

    void setSizeFrameRateAndZoom(const VideoPresetConstraints&) override;

    virtual void applyFrameRateAndZoomWithPreset(double, double, std::optional<VideoPreset>&&);
    virtual bool canResizeVideoFrames() const { return false; }

    void setSupportedPresets(Vector<VideoPreset>&&);
    void setSupportedPresets(Vector<VideoPresetData>&&);
    virtual const Vector<VideoPreset>& presets();

    bool frameRateRangeIncludesRate(const FrameRateRange&, double);

    void updateCapabilities(RealtimeMediaSourceCapabilities&);

    void dispatchVideoFrameToObservers(VideoFrame&, VideoFrameTimeMetadata);

    static std::span<const IntSize> standardVideoSizes();

    virtual Ref<TakePhotoNativePromise> takePhotoInternal(PhotoSettings&&);
    bool mutedForPhotoCapture() const { return m_mutedForPhotoCapture; }

    bool canBePowerEfficient();

private:
    struct CaptureSizeFrameRateAndZoom {
        std::optional<VideoPreset> encodingPreset;
        IntSize requestedSize;
        double requestedFrameRate { 0 };
        double requestedZoom { 0 };
    };
    bool supportsCaptureSize(std::optional<int>, std::optional<int>, const Function<bool(const IntSize&)>&&);

    enum class TryPreservingSize { No, Yes };
    std::optional<CaptureSizeFrameRateAndZoom> bestSupportedSizeFrameRateAndZoom(const VideoPresetConstraints&, TryPreservingSize = TryPreservingSize::Yes);
    std::optional<CaptureSizeFrameRateAndZoom> bestSupportedSizeFrameRateAndZoomConsideringObservers(const VideoPresetConstraints&);

    bool presetSupportsFrameRate(const VideoPreset&, double);
    bool presetSupportsZoom(const VideoPreset&, double);

    void setSizeFrameRateAndZoomForPhoto(CaptureSizeFrameRateAndZoom&&);
    Ref<TakePhotoNativePromise> takePhoto(PhotoSettings&&) final;
    bool isPowerEfficient() const final;

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const override { return "RealtimeVideoCaptureSource"_s; }
#endif

    std::optional<VideoPreset> m_currentPreset;
    Vector<VideoPreset> m_presets;
    Deque<double> m_observedFrameTimeStamps;
    double m_observedFrameRate { 0 };
    bool m_mutedForPhotoCapture { false };
};

struct SizeFrameRateAndZoom {
    std::optional<int> width;
    std::optional<int> height;
    std::optional<double> frameRate;
    std::optional<double> zoom;

    String toJSONString() const;
    Ref<JSON::Object> toJSONObject() const;
};

inline void RealtimeVideoCaptureSource::applyFrameRateAndZoomWithPreset(double, double, std::optional<VideoPreset>&&)
{
}

} // namespace WebCore

namespace WTF {
template<typename Type> struct LogArgument;
template <>
struct LogArgument<WebCore::SizeFrameRateAndZoom> {
    static String toString(const WebCore::SizeFrameRateAndZoom& size)
    {
        return size.toJSONString();
    }
};
}; // namespace WTF

#endif // ENABLE(MEDIA_STREAM)


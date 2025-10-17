/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 31, 2024.
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

#if USE(LIBWEBRTC)

#include "LibWebRTCMacros.h"
#include "RealtimeMediaSource.h"

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN

#include <webrtc/api/media_stream_interface.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

#include <wtf/RetainPtr.h>

namespace WebCore {

class CaptureDevice;
class FrameRateMonitor;

class RealtimeIncomingVideoSource
    : public RealtimeMediaSource
    , private rtc::VideoSinkInterface<webrtc::VideoFrame>
    , private webrtc::ObserverInterface
    , public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<RealtimeIncomingVideoSource, WTF::DestructionThread::MainRunLoop>
{
public:
    static Ref<RealtimeIncomingVideoSource> create(rtc::scoped_refptr<webrtc::VideoTrackInterface>&&, String&&);
    ~RealtimeIncomingVideoSource();
    WTF_ABSTRACT_THREAD_SAFE_REF_COUNTED_AND_CAN_MAKE_WEAK_PTR_IMPL;

    void enableFrameRatedMonitoring();

protected:
    RealtimeIncomingVideoSource(rtc::scoped_refptr<webrtc::VideoTrackInterface>&&, String&&);

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const final { return "RealtimeIncomingVideoSource"_s; }
#endif

    static VideoFrameTimeMetadata metadataFromVideoFrame(const webrtc::VideoFrame&);

    void notifyNewFrame();

private:
    // RealtimeMediaSource API
    void startProducingData() final;
    void stopProducingData()  final;
    void settingsDidChange(OptionSet<RealtimeMediaSourceSettings::Flag>) final;

    const RealtimeMediaSourceCapabilities& capabilities() final;
    const RealtimeMediaSourceSettings& settings() final;

    bool isIncomingVideoSource() const final { return true; }

    // webrtc::ObserverInterface API
    void OnChanged() final;

    std::optional<RealtimeMediaSourceSettings> m_currentSettings;
    rtc::scoped_refptr<webrtc::VideoTrackInterface> m_videoTrack;

    double m_currentFrameRate { -1 };
    std::unique_ptr<FrameRateMonitor> m_frameRateMonitor;
#if !RELEASE_LOG_DISABLED
    bool m_enableFrameRatedMonitoringLogging { false };
    mutable RefPtr<const Logger> m_logger;
    uint64_t m_logIdentifier { 0 };
#endif
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::RealtimeIncomingVideoSource)
static bool isType(const WebCore::RealtimeMediaSource& source) { return source.isIncomingVideoSource(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // USE(LIBWEBRTC)

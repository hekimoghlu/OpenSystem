/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 12, 2024.
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
#include "RealtimeIncomingVideoSource.h"

#if USE(LIBWEBRTC)

#include "FrameRateMonitor.h"

namespace WebCore {

static RealtimeMediaSourceSupportedConstraints supportedRealtimeIncomingVideoSourceSettingConstraints()
{
    RealtimeMediaSourceSupportedConstraints constraints;
    constraints.setSupportsWidth(true);
    constraints.setSupportsHeight(true);
    constraints.setSupportsFrameRate(true);
    constraints.setSupportsAspectRatio(true);
    return constraints;
}

RealtimeIncomingVideoSource::RealtimeIncomingVideoSource(rtc::scoped_refptr<webrtc::VideoTrackInterface>&& videoTrack, String&& videoTrackId)
    : RealtimeMediaSource(CaptureDevice { WTFMove(videoTrackId), CaptureDevice::DeviceType::Camera, "remote video"_s })
    , m_videoTrack(WTFMove(videoTrack))
{
    ASSERT(m_videoTrack);

    m_currentSettings = RealtimeMediaSourceSettings { };
    m_currentSettings->setSupportedConstraints(supportedRealtimeIncomingVideoSourceSettingConstraints());

    m_videoTrack->RegisterObserver(this);

    m_frameRateMonitor = makeUnique<FrameRateMonitor>([this](auto info) {
#if RELEASE_LOG_DISABLED
        UNUSED_PARAM(this);
        UNUSED_PARAM(info);
#else
        if (!m_enableFrameRatedMonitoringLogging)
            return;

        auto frameTime = info.frameTime.secondsSinceEpoch().value();
        auto lastFrameTime = info.lastFrameTime.secondsSinceEpoch().value();
        ALWAYS_LOG_IF(loggerPtr(), LOGIDENTIFIER, "frame at ", frameTime, " previous frame was at ", lastFrameTime, ", observed frame rate is ", info.observedFrameRate, ", delay since last frame is ", (frameTime - lastFrameTime) * 1000, " ms, frame count is ", info.frameCount);
#endif
    });
}

RealtimeIncomingVideoSource::~RealtimeIncomingVideoSource()
{
    stop();
    m_videoTrack->UnregisterObserver(this);
}

void RealtimeIncomingVideoSource::enableFrameRatedMonitoring()
{
#if !RELEASE_LOG_DISABLED
    m_enableFrameRatedMonitoringLogging = true;
#endif
}

void RealtimeIncomingVideoSource::startProducingData()
{
    m_videoTrack->AddOrUpdateSink(this, rtc::VideoSinkWants());
}

void RealtimeIncomingVideoSource::stopProducingData()
{
    m_videoTrack->RemoveSink(this);
}

void RealtimeIncomingVideoSource::OnChanged()
{
    callOnMainThread([protectedThis = Ref { *this }] {
        if (protectedThis->m_videoTrack->state() == webrtc::MediaStreamTrackInterface::kEnded)
            protectedThis->end();
    });
}

const RealtimeMediaSourceCapabilities& RealtimeIncomingVideoSource::capabilities()
{
    return RealtimeMediaSourceCapabilities::emptyCapabilities();
}

const RealtimeMediaSourceSettings& RealtimeIncomingVideoSource::settings()
{
    if (m_currentSettings)
        return m_currentSettings.value();

    RealtimeMediaSourceSettings settings;
    settings.setSupportedConstraints(supportedRealtimeIncomingVideoSourceSettingConstraints());

    auto size = this->size();
    settings.setWidth(size.width());
    settings.setHeight(size.height());
    settings.setFrameRate(frameRate());

    m_currentSettings = WTFMove(settings);
    return m_currentSettings.value();
}

void RealtimeIncomingVideoSource::settingsDidChange(OptionSet<RealtimeMediaSourceSettings::Flag> settings)
{
    if (settings.containsAny({ RealtimeMediaSourceSettings::Flag::FrameRate, RealtimeMediaSourceSettings::Flag::Height, RealtimeMediaSourceSettings::Flag::Width }))
        m_currentSettings = std::nullopt;
}

VideoFrameTimeMetadata RealtimeIncomingVideoSource::metadataFromVideoFrame(const webrtc::VideoFrame& frame)
{
    VideoFrameTimeMetadata metadata;
    if (frame.ntp_time_ms() > 0)
        metadata.captureTime = Seconds::fromMilliseconds(frame.ntp_time_ms());
    if (isInBounds<uint32_t>(frame.rtp_timestamp()))
        metadata.rtpTimestamp = frame.rtp_timestamp();
    auto lastPacketTimestamp = std::max_element(frame.packet_infos().cbegin(), frame.packet_infos().cend(), [](const auto& a, const auto& b) {
        return a.receive_time() < b.receive_time();
    });
    metadata.receiveTime = Seconds::fromMicroseconds(lastPacketTimestamp->receive_time().us());
    if (frame.processing_time())
        metadata.processingDuration = Seconds::fromMilliseconds(frame.processing_time()->Elapsed().ms()).value();

    return metadata;
}

void RealtimeIncomingVideoSource::notifyNewFrame()
{
    if (!m_frameRateMonitor)
        return;

    m_frameRateMonitor->update();

    auto observedFrameRate = m_frameRateMonitor->observedFrameRate();
    if (m_currentFrameRate > 0 && fabs(m_currentFrameRate - observedFrameRate) < 1)
        return;

    m_currentFrameRate = observedFrameRate;
    callOnMainThread([protectedThis = Ref { *this }, observedFrameRate] {
        protectedThis->setFrameRate(observedFrameRate);
    });
}

} // namespace WebCore

#endif // USE(LIBWEBRTC)

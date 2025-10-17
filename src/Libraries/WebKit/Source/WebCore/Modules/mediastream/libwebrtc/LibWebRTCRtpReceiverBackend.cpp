/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 2, 2023.
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
#include "LibWebRTCRtpReceiverBackend.h"

#if ENABLE(WEB_RTC) && USE(LIBWEBRTC)

#include "Document.h"
#include "LibWebRTCAudioModule.h"
#include "LibWebRTCDtlsTransportBackend.h"
#include "LibWebRTCProvider.h"
#include "LibWebRTCRtpReceiverTransformBackend.h"
#include "LibWebRTCUtils.h"
#include "Page.h"
#include "RTCRtpTransformBackend.h"
#include "RealtimeIncomingAudioSource.h"
#include "RealtimeIncomingVideoSource.h"
#include <wtf/TZoneMallocInlines.h>

ALLOW_UNUSED_PARAMETERS_BEGIN
ALLOW_DEPRECATED_DECLARATIONS_BEGIN

#include <webrtc/api/rtp_receiver_interface.h>

ALLOW_DEPRECATED_DECLARATIONS_END
ALLOW_UNUSED_PARAMETERS_END

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LibWebRTCRtpReceiverBackend);

LibWebRTCRtpReceiverBackend::LibWebRTCRtpReceiverBackend(rtc::scoped_refptr<webrtc::RtpReceiverInterface>&& rtcReceiver)
    : m_rtcReceiver(WTFMove(rtcReceiver))
{
}

LibWebRTCRtpReceiverBackend::~LibWebRTCRtpReceiverBackend() = default;

RTCRtpParameters LibWebRTCRtpReceiverBackend::getParameters()
{
    return toRTCRtpParameters(m_rtcReceiver->GetParameters());
}

static inline void fillRTCRtpContributingSource(RTCRtpContributingSource& source, const webrtc::RtpSource& rtcSource)
{
    source.timestamp = rtcSource.timestamp().ms();
    source.rtpTimestamp = rtcSource.rtp_timestamp();
    source.source = rtcSource.source_id();
    if (rtcSource.audio_level())
        source.audioLevel = (*rtcSource.audio_level() == 127) ? 0 : pow(10, -*rtcSource.audio_level() / 20);
}

static inline RTCRtpContributingSource toRTCRtpContributingSource(const webrtc::RtpSource& rtcSource)
{
    RTCRtpContributingSource source;
    fillRTCRtpContributingSource(source, rtcSource);
    return source;
}

static inline RTCRtpSynchronizationSource toRTCRtpSynchronizationSource(const webrtc::RtpSource& rtcSource)
{
    RTCRtpSynchronizationSource source;
    fillRTCRtpContributingSource(source, rtcSource);
    return source;
}

Vector<RTCRtpContributingSource> LibWebRTCRtpReceiverBackend::getContributingSources() const
{
    Vector<RTCRtpContributingSource> sources;
    for (auto& rtcSource : m_rtcReceiver->GetSources()) {
        if (rtcSource.source_type() == webrtc::RtpSourceType::CSRC)
            sources.append(toRTCRtpContributingSource(rtcSource));
    }
    return sources;
}

Vector<RTCRtpSynchronizationSource> LibWebRTCRtpReceiverBackend::getSynchronizationSources() const
{
    Vector<RTCRtpSynchronizationSource> sources;
    for (auto& rtcSource : m_rtcReceiver->GetSources()) {
        if (rtcSource.source_type() == webrtc::RtpSourceType::SSRC)
            sources.append(toRTCRtpSynchronizationSource(rtcSource));
    }
    return sources;
}

Ref<RealtimeMediaSource> LibWebRTCRtpReceiverBackend::createSource(Document& document)
{
    auto rtcTrack = m_rtcReceiver->track();
    switch (m_rtcReceiver->media_type()) {
    case cricket::MEDIA_TYPE_DATA:
    case cricket::MEDIA_TYPE_UNSUPPORTED:
        break;
    case cricket::MEDIA_TYPE_AUDIO: {
        rtc::scoped_refptr<webrtc::AudioTrackInterface> audioTrack { static_cast<webrtc::AudioTrackInterface*>(rtcTrack.get()) };
        auto source = RealtimeIncomingAudioSource::create(WTFMove(audioTrack), fromStdString(rtcTrack->id()));
        if (document.page()) {
            auto& webRTCProvider = reinterpret_cast<LibWebRTCProvider&>(document.page()->webRTCProvider());
            source->setAudioModule(webRTCProvider.audioModule());
        }
        return source;
    }
    case cricket::MEDIA_TYPE_VIDEO: {
        rtc::scoped_refptr<webrtc::VideoTrackInterface> videoTrack { static_cast<webrtc::VideoTrackInterface*>(rtcTrack.get()) };
        auto source = RealtimeIncomingVideoSource::create(WTFMove(videoTrack), fromStdString(rtcTrack->id()));
        if (document.settings().webRTCMediaPipelineAdditionalLoggingEnabled())
            source->enableFrameRatedMonitoring();
        return source;
    }
    }
    RELEASE_ASSERT_NOT_REACHED();
}

Ref<RTCRtpTransformBackend> LibWebRTCRtpReceiverBackend::rtcRtpTransformBackend()
{
    if (!m_transformBackend)
        m_transformBackend = LibWebRTCRtpReceiverTransformBackend::create(m_rtcReceiver);
    return *m_transformBackend;
}

std::unique_ptr<RTCDtlsTransportBackend> LibWebRTCRtpReceiverBackend::dtlsTransportBackend()
{
    auto backend = m_rtcReceiver->dtls_transport();
    return backend ? makeUnique<LibWebRTCDtlsTransportBackend>(WTFMove(backend)) : nullptr;
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(LIBWEBRTC)

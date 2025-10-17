/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 27, 2023.
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
#include "LibWebRTCRtpTransceiverBackend.h"

#if ENABLE(WEB_RTC) && USE(LIBWEBRTC)

#include "JSDOMPromiseDeferred.h"
#include "LibWebRTCRtpReceiverBackend.h"
#include "LibWebRTCRtpSenderBackend.h"
#include "LibWebRTCUtils.h"
#include "RTCRtpCodecCapability.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LibWebRTCRtpTransceiverBackend);

std::unique_ptr<LibWebRTCRtpReceiverBackend> LibWebRTCRtpTransceiverBackend::createReceiverBackend()
{
    return makeUnique<LibWebRTCRtpReceiverBackend>(m_rtcTransceiver->receiver());
}

std::unique_ptr<LibWebRTCRtpSenderBackend> LibWebRTCRtpTransceiverBackend::createSenderBackend(LibWebRTCPeerConnectionBackend& backend, LibWebRTCRtpSenderBackend::Source&& source)
{
    return makeUnique<LibWebRTCRtpSenderBackend>(backend, m_rtcTransceiver->sender(), WTFMove(source));
}

RTCRtpTransceiverDirection LibWebRTCRtpTransceiverBackend::direction() const
{
    return toRTCRtpTransceiverDirection(m_rtcTransceiver->direction());
}

std::optional<RTCRtpTransceiverDirection> LibWebRTCRtpTransceiverBackend::currentDirection() const
{
    auto value = m_rtcTransceiver->current_direction();
    if (!value)
        return std::nullopt;
    return toRTCRtpTransceiverDirection(*value);
}

void LibWebRTCRtpTransceiverBackend::setDirection(RTCRtpTransceiverDirection direction)
{
    // FIXME: Handle error.
    m_rtcTransceiver->SetDirectionWithError(fromRTCRtpTransceiverDirection(direction));
}

String LibWebRTCRtpTransceiverBackend::mid()
{
    if (auto mid = m_rtcTransceiver->mid())
        return fromStdString(*mid);
    return String { };
}

void LibWebRTCRtpTransceiverBackend::stop()
{
    m_rtcTransceiver->StopStandard();
}

bool LibWebRTCRtpTransceiverBackend::stopped() const
{
    return m_rtcTransceiver->stopped();
}

static inline ExceptionOr<webrtc::RtpCodecCapability> toRtpCodecCapability(const RTCRtpCodecCapability& codec)
{
    webrtc::RtpCodecCapability rtcCodec;
    if (codec.mimeType.startsWith("video/"_s))
        rtcCodec.kind = cricket::MEDIA_TYPE_VIDEO;
    else if (codec.mimeType.startsWith("audio/"_s))
        rtcCodec.kind = cricket::MEDIA_TYPE_AUDIO;
    else
        return Exception { ExceptionCode::InvalidModificationError, "RTCRtpCodecCapability bad mimeType"_s };

    rtcCodec.name = StringView(codec.mimeType).substring(6).utf8().toStdString();
    rtcCodec.clock_rate = codec.clockRate;
    if (codec.channels)
        rtcCodec.num_channels = *codec.channels;

    for (auto parameter : StringView(codec.sdpFmtpLine).split(';')) {
        auto position = parameter.find('=');
        if (position == notFound)
            return Exception { ExceptionCode::InvalidModificationError, "RTCRtpCodecCapability sdpFmtLine badly formated"_s };
        rtcCodec.parameters.emplace(parameter.left(position).utf8().data(), parameter.substring(position + 1).utf8().data());
    }

    return rtcCodec;
}

ExceptionOr<void> LibWebRTCRtpTransceiverBackend::setCodecPreferences(const Vector<RTCRtpCodecCapability>& codecs)
{
    std::vector<webrtc::RtpCodecCapability> rtcCodecs;
    for (auto& codec : codecs) {
        auto result = toRtpCodecCapability(codec);
        if (result.hasException())
            return result.releaseException();
        rtcCodecs.push_back(result.releaseReturnValue());
    }
    auto result = m_rtcTransceiver->SetCodecPreferences(rtcCodecs);
    if (!result.ok())
        return toException(result);
    return { };
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(LIBWEBRTC)

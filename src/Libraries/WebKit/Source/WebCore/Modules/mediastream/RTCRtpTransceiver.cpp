/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 27, 2021.
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
#include "RTCRtpTransceiver.h"

#if ENABLE(WEB_RTC)

#include "RTCPeerConnection.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RTCRtpTransceiver);

RTCRtpTransceiver::RTCRtpTransceiver(Ref<RTCRtpSender>&& sender, Ref<RTCRtpReceiver>&& receiver, std::unique_ptr<RTCRtpTransceiverBackend>&& backend)
    : m_direction(RTCRtpTransceiverDirection::Sendrecv)
    , m_sender(WTFMove(sender))
    , m_receiver(WTFMove(receiver))
    , m_backend(WTFMove(backend))
{
}

RTCRtpTransceiver::~RTCRtpTransceiver() = default;

String RTCRtpTransceiver::mid() const
{
    return m_backend ? m_backend->mid() : String { };
}

bool RTCRtpTransceiver::hasSendingDirection() const
{
    return m_direction == RTCRtpTransceiverDirection::Sendrecv || m_direction == RTCRtpTransceiverDirection::Sendonly;
}

RTCRtpTransceiverDirection RTCRtpTransceiver::direction() const
{
    if (!m_backend)
        return m_direction;
    return m_backend->direction();
}

std::optional<RTCRtpTransceiverDirection> RTCRtpTransceiver::currentDirection() const
{
    if (!m_backend)
        return std::nullopt;
    return m_backend->currentDirection();
}

void RTCRtpTransceiver::setDirection(RTCRtpTransceiverDirection direction)
{
    m_direction = direction;
    if (m_backend)
        m_backend->setDirection(direction);
}


void RTCRtpTransceiver::enableSendingDirection()
{
    if (m_direction == RTCRtpTransceiverDirection::Recvonly)
        m_direction = RTCRtpTransceiverDirection::Sendrecv;
    else if (m_direction == RTCRtpTransceiverDirection::Inactive)
        m_direction = RTCRtpTransceiverDirection::Sendonly;
}

void RTCRtpTransceiver::disableSendingDirection()
{
    if (m_direction == RTCRtpTransceiverDirection::Sendrecv)
        m_direction = RTCRtpTransceiverDirection::Recvonly;
    else if (m_direction == RTCRtpTransceiverDirection::Sendonly)
        m_direction = RTCRtpTransceiverDirection::Inactive;
}

void RTCRtpTransceiver::setConnection(RTCPeerConnection& connection)
{
    ASSERT(!m_connection);
    m_connection = connection;
}

ExceptionOr<void> RTCRtpTransceiver::stop()
{
    if (!m_connection || m_connection->isClosed())
        return Exception { ExceptionCode::InvalidStateError, "RTCPeerConnection is closed"_s };

    if (m_stopped)
        return { };

    m_stopped = true;
    m_receiver->stop();
    m_sender->stop();
    if (m_backend)
        m_backend->stop();

    // No need to call negotiation needed, it will be done by the backend itself.
    return { };
}

ExceptionOr<void> RTCRtpTransceiver::setCodecPreferences(const Vector<RTCRtpCodecCapability>& codecs)
{
    if (!m_backend)
        return { };

    RELEASE_LOG_INFO(WebRTC, "RTCRtpTransceiver::setCodecPreferences");
    return m_backend->setCodecPreferences(codecs);
}

bool RTCRtpTransceiver::stopped() const
{
    if (m_backend)
        return m_backend->stopped();
    return m_stopped;
}

void RtpTransceiverSet::append(Ref<RTCRtpTransceiver>&& transceiver)
{
    m_transceivers.append(WTFMove(transceiver));
}

Vector<std::reference_wrapper<RTCRtpSender>> RtpTransceiverSet::senders() const
{
    Vector<std::reference_wrapper<RTCRtpSender>> senders;
    for (auto& transceiver : m_transceivers) {
        if (transceiver->stopped())
            continue;
        senders.append(transceiver->sender());
    }
    return senders;
}

Vector<std::reference_wrapper<RTCRtpReceiver>> RtpTransceiverSet::receivers() const
{
    Vector<std::reference_wrapper<RTCRtpReceiver>> receivers;
    for (auto& transceiver : m_transceivers) {
        if (transceiver->stopped())
            continue;
        receivers.append(transceiver->receiver());
    }
    return receivers;
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC)

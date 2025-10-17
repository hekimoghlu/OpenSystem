/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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

#if ENABLE(WEB_RTC)

#include "RTCRtpReceiver.h"
#include "RTCRtpSender.h"
#include "RTCRtpTransceiverBackend.h"
#include "RTCRtpTransceiverDirection.h"
#include "ScriptWrappable.h"
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class RTCPeerConnection;
struct RTCRtpCodecCapability;

class RTCRtpTransceiver final : public RefCounted<RTCRtpTransceiver>, public ScriptWrappable {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RTCRtpTransceiver);
public:
    static Ref<RTCRtpTransceiver> create(Ref<RTCRtpSender>&& sender, Ref<RTCRtpReceiver>&& receiver, std::unique_ptr<RTCRtpTransceiverBackend>&& backend) { return adoptRef(*new RTCRtpTransceiver(WTFMove(sender), WTFMove(receiver), WTFMove(backend))); }
    virtual ~RTCRtpTransceiver();

    bool hasSendingDirection() const;
    void enableSendingDirection();
    void disableSendingDirection();

    RTCRtpTransceiverDirection direction() const;
    std::optional<RTCRtpTransceiverDirection> currentDirection() const;
    void setDirection(RTCRtpTransceiverDirection);
    String mid() const;

    RTCRtpSender& sender() { return m_sender.get(); }
    RTCRtpReceiver& receiver() { return m_receiver.get(); }

    bool stopped() const;
    ExceptionOr<void> stop();
    ExceptionOr<void> setCodecPreferences(const Vector<RTCRtpCodecCapability>&);

    RTCRtpTransceiverBackend* backend() { return m_backend.get(); }
    void setConnection(RTCPeerConnection&);

    std::optional<RTCRtpTransceiverDirection> firedDirection() const { return m_firedDirection; }
    void setFiredDirection(std::optional<RTCRtpTransceiverDirection> firedDirection) { m_firedDirection = firedDirection; }

private:
    RTCRtpTransceiver(Ref<RTCRtpSender>&&, Ref<RTCRtpReceiver>&&, std::unique_ptr<RTCRtpTransceiverBackend>&&);

    RTCRtpTransceiverDirection m_direction;
    std::optional<RTCRtpTransceiverDirection> m_firedDirection;

    Ref<RTCRtpSender> m_sender;
    Ref<RTCRtpReceiver> m_receiver;

    bool m_stopped { false };

    std::unique_ptr<RTCRtpTransceiverBackend> m_backend;
    WeakPtr<RTCPeerConnection, WeakPtrImplWithEventTargetData> m_connection;
};

class RtpTransceiverSet {
public:
    const Vector<RefPtr<RTCRtpTransceiver>>& list() const { return m_transceivers; }
    void append(Ref<RTCRtpTransceiver>&&);

    Vector<std::reference_wrapper<RTCRtpSender>> senders() const;
    Vector<std::reference_wrapper<RTCRtpReceiver>> receivers() const;

private:
    Vector<RefPtr<RTCRtpTransceiver>> m_transceivers;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)

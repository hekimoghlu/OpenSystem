/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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

#include "ActiveDOMObject.h"
#include "DetachedRTCDataChannel.h"
#include "Event.h"
#include "EventTarget.h"
#include "ExceptionOr.h"
#include "NetworkSendQueue.h"
#include "RTCDataChannelHandler.h"
#include "RTCDataChannelHandlerClient.h"
#include "RTCDataChannelIdentifier.h"
#include "ScriptExecutionContext.h"
#include "ScriptWrappable.h"
#include "Timer.h"

namespace JSC {
    class ArrayBuffer;
    class ArrayBufferView;
}

namespace WebCore {

class Blob;
class RTCPeerConnectionHandler;

class RTCDataChannel final : public RefCounted<RTCDataChannel>, public ActiveDOMObject, public RTCDataChannelHandlerClient, public EventTarget {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RTCDataChannel);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<RTCDataChannel> create(ScriptExecutionContext&, std::unique_ptr<RTCDataChannelHandler>&&, String&&, RTCDataChannelInit&&, RTCDataChannelState);
    static Ref<RTCDataChannel> create(ScriptExecutionContext&, RTCDataChannelIdentifier, String&&, RTCDataChannelInit&&, RTCDataChannelState);

    bool ordered() const { return *m_options.ordered; }
    std::optional<unsigned short> maxPacketLifeTime() const { return m_options.maxPacketLifeTime; }
    std::optional<unsigned short> maxRetransmits() const { return m_options.maxRetransmits; }
    String protocol() const { return m_options.protocol; }
    bool negotiated() const { return *m_options.negotiated; };
    std::optional<unsigned short> id() const;
    RTCPriorityType priority() const { return m_options.priority; };
    const RTCDataChannelInit& options() const { return m_options; }

    const String& label() const { return m_label; }
    RTCDataChannelState readyState() const {return m_readyState; }
    size_t bufferedAmount() const final { return m_bufferedAmount; }
    size_t bufferedAmountLowThreshold() const { return m_bufferedAmountLowThreshold; }
    void setBufferedAmountLowThreshold(size_t value) { m_bufferedAmountLowThreshold = value; }

    enum class BinaryType : bool { Blob, Arraybuffer };
    BinaryType binaryType() const { return m_binaryType; }
    void setBinaryType(BinaryType);

    ExceptionOr<void> send(const String&);
    ExceptionOr<void> send(JSC::ArrayBuffer&);
    ExceptionOr<void> send(JSC::ArrayBufferView&);
    ExceptionOr<void> send(Blob&);

    void close();

    RTCDataChannelIdentifier identifier() const { return m_identifier; }
    bool canDetach() const;
    std::unique_ptr<DetachedRTCDataChannel> detach();

    WEBCORE_EXPORT static std::unique_ptr<RTCDataChannelHandler> handlerFromIdentifier(RTCDataChannelLocalIdentifier);
    void fireOpenEventIfNeeded();

private:
    RTCDataChannel(ScriptExecutionContext&, std::unique_ptr<RTCDataChannelHandler>&&, String&&, RTCDataChannelInit&&, RTCDataChannelState);

    static NetworkSendQueue createMessageQueue(ScriptExecutionContext&, RTCDataChannel&);

    void scheduleDispatchEvent(Ref<Event>&&);
    void removeFromDataChannelLocalMapIfNeeded();

    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::RTCDataChannel; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }

    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    // ActiveDOMObject.
    void stop() final;
    bool virtualHasPendingActivity() const final;

    // RTCDataChannelHandlerClient API
    void didChangeReadyState(RTCDataChannelState) final;
    void didReceiveStringData(const String&) final;
    void didReceiveRawData(std::span<const uint8_t>) final;
    void didDetectError(Ref<RTCError>&&) final;
    void bufferedAmountIsDecreasing(size_t) final;

    std::unique_ptr<RTCDataChannelHandler> m_handler;
    RTCDataChannelIdentifier m_identifier;
    Markable<ScriptExecutionContextIdentifier> m_contextIdentifier;
    // FIXME: m_stopped is probably redundant with m_readyState.
    bool m_stopped { false };
    RTCDataChannelState m_readyState { RTCDataChannelState::Connecting };

    BinaryType m_binaryType { BinaryType::Arraybuffer };

    String m_label;
    RTCDataChannelInit m_options;
    size_t m_bufferedAmount { 0 };
    size_t m_bufferedAmountLowThreshold { 0 };
    bool m_isDetachable { true };
    bool m_isDetached { false };
    NetworkSendQueue m_messageQueue;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)

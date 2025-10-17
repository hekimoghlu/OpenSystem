/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 24, 2024.
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

#if ENABLE(WEB_RTC) && USE(LIBWEBRTC)

#include "LibWebRTCMacros.h"
#include "ProcessQualified.h"
#include "RTCDataChannelHandler.h"
#include "RTCDataChannelState.h"
#include "SharedBuffer.h"
#include <wtf/Lock.h>
#include <wtf/ObjectIdentifier.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN

#include <webrtc/api/data_channel_interface.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace webrtc {
struct DataChannelInit;
}

namespace WebCore {

class Document;
class RTCDataChannelEvent;
class RTCDataChannelHandlerClient;
struct RTCDataChannelInit;
class ScriptExecutionContext;

class LibWebRTCDataChannelHandler final : public RTCDataChannelHandler, private webrtc::DataChannelObserver {
    WTF_MAKE_TZONE_ALLOCATED(LibWebRTCDataChannelHandler);
public:
    explicit LibWebRTCDataChannelHandler(rtc::scoped_refptr<webrtc::DataChannelInterface>&&);
    ~LibWebRTCDataChannelHandler();

    RTCDataChannelInit dataChannelInit() const;
    String label() const;

    static webrtc::DataChannelInit fromRTCDataChannelInit(const RTCDataChannelInit&);

private:
    // RTCDataChannelHandler API
    void setClient(RTCDataChannelHandlerClient&, std::optional<ScriptExecutionContextIdentifier>) final;
    bool sendStringData(const CString&) final;
    bool sendRawData(std::span<const uint8_t>) final;
    void close() final;
    std::optional<unsigned short> id() const final;

    // webrtc::DataChannelObserver API
    void OnStateChange();
    void OnMessage(const webrtc::DataBuffer&);
    void OnBufferedAmountChange(uint64_t);

    void checkState();

    struct StateChange {
        RTCDataChannelState state;
        std::optional<webrtc::RTCError> error;
    };
    using Message = std::variant<StateChange, String, Ref<FragmentedSharedBuffer>>;
    using PendingMessages = Vector<Message>;
    void storeMessage(PendingMessages&, const webrtc::DataBuffer&);
    void processMessage(const webrtc::DataBuffer&);
    void processStoredMessage(Message&);

    void postTask(Function<void()>&&);

    rtc::scoped_refptr<webrtc::DataChannelInterface> m_channel;
    Lock m_clientLock;
    bool m_hasClient WTF_GUARDED_BY_LOCK(m_clientLock)  { false };
    WeakPtr<RTCDataChannelHandlerClient> m_client WTF_GUARDED_BY_LOCK(m_clientLock) { nullptr };
    Markable<ScriptExecutionContextIdentifier> m_contextIdentifier;
    PendingMessages m_bufferedMessages WTF_GUARDED_BY_LOCK(m_clientLock);
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(LIBWEBRTC)

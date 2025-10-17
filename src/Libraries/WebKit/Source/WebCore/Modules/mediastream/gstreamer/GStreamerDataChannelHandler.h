/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 27, 2022.
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

#if USE(GSTREAMER_WEBRTC)

#include "GRefPtrGStreamer.h"
#include "GUniquePtrGStreamer.h"
#include "RTCDataChannelHandler.h"
#include "RTCDataChannelState.h"
#include "SharedBuffer.h"

#include <wtf/Condition.h>
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Document;
class RTCDataChannelEvent;
class RTCDataChannelHandlerClient;
struct RTCDataChannelInit;

class GStreamerDataChannelHandler final : public RTCDataChannelHandler {
    WTF_MAKE_TZONE_ALLOCATED(GStreamerDataChannelHandler);
public:
    explicit GStreamerDataChannelHandler(GRefPtr<GstWebRTCDataChannel>&&);
    ~GStreamerDataChannelHandler();

    RTCDataChannelInit dataChannelInit() const;
    String label() const;

    static GUniquePtr<GstStructure> fromRTCDataChannelInit(const RTCDataChannelInit&);

    const GstWebRTCDataChannel* channel() const { return m_channel.get(); }

private:
    // RTCDataChannelHandler API
    void setClient(RTCDataChannelHandlerClient&, std::optional<ScriptExecutionContextIdentifier>) final;
    bool sendStringData(const CString&) final;
    bool sendRawData(std::span<const uint8_t>) final;
    std::optional<unsigned short> id() const final;
    void close() final;

    void onMessageData(GBytes*);
    void onMessageString(const char*);
    void onError(GError*);
    void onClose();

    void readyStateChanged();
    void bufferedAmountChanged();
    bool checkState();
    void postTask(Function<void()>&&);

    struct StateChange {
        RTCDataChannelState state;
        std::optional<GError*> error;
    };
    using Message = std::variant<StateChange, String, Ref<FragmentedSharedBuffer>>;
    using PendingMessages = Vector<Message>;

    Lock m_clientLock;
    GRefPtr<GstWebRTCDataChannel> m_channel;
    std::optional<WeakPtr<RTCDataChannelHandlerClient>> m_client WTF_GUARDED_BY_LOCK(m_clientLock);
    Markable<ScriptExecutionContextIdentifier> m_contextIdentifier;
    PendingMessages m_pendingMessages WTF_GUARDED_BY_LOCK(m_clientLock);

    std::optional<size_t> m_cachedBufferedAmount;
    bool m_closing { false };

    String m_channelId;
};

} // namespace WebCore

#endif // USE(GSTREAMER_WEBRTC)

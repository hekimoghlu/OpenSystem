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
#pragma once

#if ENABLE(WEB_RTC)

#include "RTCDataChannelHandler.h"
#include "RTCDataChannelIdentifier.h"
#include "RTCDataChannelState.h"
#include <wtf/Function.h>
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class RTCDataChannelRemoteHandler;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::RTCDataChannelRemoteHandler> : std::true_type { };
}

namespace WebCore {

class RTCDataChannelHandlerClient;
class RTCDataChannelRemoteHandlerConnection;
class RTCError;
class FragmentedSharedBuffer;

class RTCDataChannelRemoteHandler final : public RTCDataChannelHandler, public CanMakeWeakPtr<RTCDataChannelRemoteHandler> {
    WTF_MAKE_TZONE_ALLOCATED(RTCDataChannelRemoteHandler);
public:
    static std::unique_ptr<RTCDataChannelRemoteHandler> create(RTCDataChannelIdentifier, RefPtr<RTCDataChannelRemoteHandlerConnection>&&);
    RTCDataChannelRemoteHandler(RTCDataChannelIdentifier, Ref<RTCDataChannelRemoteHandlerConnection>&&);
    ~RTCDataChannelRemoteHandler();

    WEBCORE_EXPORT void didChangeReadyState(RTCDataChannelState);
    WEBCORE_EXPORT void didReceiveStringData(String&&);
    WEBCORE_EXPORT void didReceiveRawData(std::span<const uint8_t>);
    WEBCORE_EXPORT void didDetectError(Ref<RTCError>&&);
    WEBCORE_EXPORT void bufferedAmountIsDecreasing(size_t);

    WEBCORE_EXPORT void readyToSend();

    void setLocalIdentifier(RTCDataChannelIdentifier localIdentifier) { m_localIdentifier = localIdentifier; }

private:
    // RTCDataChannelHandler
    void setClient(RTCDataChannelHandlerClient&, std::optional<ScriptExecutionContextIdentifier>) final;
    bool sendStringData(const CString&) final;
    bool sendRawData(std::span<const uint8_t>) final;
    void close() final;

    RTCDataChannelIdentifier m_remoteIdentifier;
    Markable<RTCDataChannelIdentifier> m_localIdentifier;

    RTCDataChannelHandlerClient* m_client { nullptr };
    Ref<RTCDataChannelRemoteHandlerConnection> m_connection;

    struct Message {
        bool isRaw { false };
        Ref<FragmentedSharedBuffer> buffer;
    };
    Vector<Message> m_pendingMessages;
    bool m_isPendingClose { false };
    bool m_isReadyToSend { false };

};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)

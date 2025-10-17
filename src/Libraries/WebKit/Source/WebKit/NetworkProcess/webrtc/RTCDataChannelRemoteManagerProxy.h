/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 28, 2023.
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

#include "Connection.h"
#include "WorkQueueMessageReceiver.h"
#include <WebCore/RTCDataChannelRemoteHandlerConnection.h>
#include <WebCore/RTCDataChannelRemoteSourceConnection.h>
#include <wtf/WorkQueue.h>

namespace WebKit {

class NetworkConnectionToWebProcess;

class RTCDataChannelRemoteManagerProxy final : public IPC::WorkQueueMessageReceiver {
public:
    static Ref<RTCDataChannelRemoteManagerProxy> create() { return adoptRef(*new RTCDataChannelRemoteManagerProxy); }

    void registerConnectionToWebProcess(NetworkConnectionToWebProcess&);
    void unregisterConnectionToWebProcess(NetworkConnectionToWebProcess&);

private:
    RTCDataChannelRemoteManagerProxy();

    Ref<WorkQueue> protectedQueue();

    // IPC::WorkQueueMessageReceiver overrides.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    // To source
    void sendData(WebCore::RTCDataChannelIdentifier, bool isRaw, std::span<const uint8_t>);
    void close(WebCore::RTCDataChannelIdentifier);

    // To handler
    void changeReadyState(WebCore::RTCDataChannelIdentifier, WebCore::RTCDataChannelState);
    void receiveData(WebCore::RTCDataChannelIdentifier, bool isRaw, std::span<const uint8_t>);
    void detectError(WebCore::RTCDataChannelIdentifier, WebCore::RTCErrorDetailType, const String&);
    void bufferedAmountIsDecreasing(WebCore::RTCDataChannelIdentifier, size_t amount);

    Ref<WorkQueue> m_queue;
    HashMap<WebCore::ProcessIdentifier, IPC::Connection::UniqueID> m_webProcessConnections;
};

} // namespace WebKit

#endif // ENABLE(WEB_RTC)

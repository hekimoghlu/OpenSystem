/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 9, 2021.
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

#include "DaemonConnection.h"
#include "PrivateClickMeasurementManagerInterface.h"

namespace WebKit {

class NetworkSession;

namespace PCM {

enum class MessageType : uint8_t;

struct ConnectionTraits {
    using MessageType = WebKit::PCM::MessageType;
    static constexpr auto protocolVersionKey { PCM::protocolVersionKey };
    static constexpr uint64_t protocolVersionValue { PCM::protocolVersionValue };
    static constexpr auto protocolEncodedMessageKey { PCM::protocolEncodedMessageKey };
};

class Connection final : public Daemon::ConnectionToMachService<ConnectionTraits> {
public:
    static Ref<Connection> create(CString&& machServiceName, NetworkSession&);

private:
    Connection(CString&& machServiceName, NetworkSession&);

    void newConnectionWasInitialized() const final;
#if PLATFORM(COCOA)
    OSObjectPtr<xpc_object_t> dictionaryFromMessage(MessageType, Daemon::EncodedMessage&&) const final;
    void connectionReceivedEvent(xpc_object_t) final;
#endif

    WeakPtr<NetworkSession> m_networkSession;
};

} // namespace PCM

} // namespace WebKit

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 4, 2022.
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

#if ENABLE(REMOTE_INSPECTOR)

#include "RemoteControllableTarget.h"
#include "RemoteInspectorMessageParser.h"
#include "RemoteInspectorSocketEndpoint.h"
#include <wtf/HashMap.h>
#include <wtf/JSONValues.h>
#include <wtf/Lock.h>
#include <wtf/text/WTFString.h>

namespace Inspector {

class MessageParser;

class RemoteInspectorConnectionClient : public RemoteInspectorSocketEndpoint::Client {
public:
    JS_EXPORT_PRIVATE ~RemoteInspectorConnectionClient() override;

    JS_EXPORT_PRIVATE std::optional<ConnectionID> connectInet(const char* serverAddr, uint16_t serverPort);
    std::optional<ConnectionID> createClient(PlatformSocketType);
    JS_EXPORT_PRIVATE void send(ConnectionID, std::span<const uint8_t>);

    JS_EXPORT_PRIVATE void didReceive(RemoteInspectorSocketEndpoint&, ConnectionID, Vector<uint8_t>&&) final;

    struct Event {
        String methodName;
        ConnectionID clientID { };
        std::optional<ConnectionID> connectionID;
        std::optional<TargetID> targetID;
        std::optional<String> message;
    };

    using CallHandler = void (RemoteInspectorConnectionClient::*)(const Event&);
    virtual HashMap<String, CallHandler>& dispatchMap() = 0;

protected:
    JS_EXPORT_PRIVATE std::optional<Vector<Ref<JSON::Object>>> parseTargetListJSON(const String&);

    static std::optional<Event> extractEvent(ConnectionID, Vector<uint8_t>&&);

    HashMap<ConnectionID, MessageParser> m_parsers WTF_GUARDED_BY_LOCK(m_parsersLock);
    Lock m_parsersLock;
};

} // namespace Inspector

#endif // ENABLE(REMOTE_INSPECTOR)

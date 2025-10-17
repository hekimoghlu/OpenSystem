/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 28, 2022.
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
#include "RemoteInspectorConnectionClient.h"

#if ENABLE(REMOTE_INSPECTOR)

#include "RemoteInspectorSocketEndpoint.h"
#include <wtf/JSONValues.h>
#include <wtf/RunLoop.h>

namespace Inspector {

RemoteInspectorConnectionClient::~RemoteInspectorConnectionClient()
{
    auto& endpoint = Inspector::RemoteInspectorSocketEndpoint::singleton();
    endpoint.invalidateClient(*this);
}

std::optional<ConnectionID> RemoteInspectorConnectionClient::connectInet(const char* serverAddr, uint16_t serverPort)
{
    auto& endpoint = Inspector::RemoteInspectorSocketEndpoint::singleton();
    return endpoint.connectInet(serverAddr, serverPort, *this);
}

std::optional<ConnectionID> RemoteInspectorConnectionClient::createClient(PlatformSocketType socket)
{
    auto& endpoint = Inspector::RemoteInspectorSocketEndpoint::singleton();
    return endpoint.createClient(socket, *this);
}

void RemoteInspectorConnectionClient::send(ConnectionID id, std::span<const uint8_t> data)
{
    auto message = MessageParser::createMessage(data);
    if (message.isEmpty())
        return;

    auto& endpoint = RemoteInspectorSocketEndpoint::singleton();
    endpoint.send(id, message);
}

void RemoteInspectorConnectionClient::didReceive(RemoteInspectorSocketEndpoint&, ConnectionID clientID, Vector<uint8_t>&& data)
{
    ASSERT(!isMainThread());

    Locker locker { m_parsersLock };
    auto result = m_parsers.ensure(clientID, [this, clientID] {
        return MessageParser([this, clientID](Vector<uint8_t>&& data) {
            if (auto event = RemoteInspectorConnectionClient::extractEvent(clientID, WTFMove(data))) {
                RunLoop::main().dispatch([this, event = WTFMove(*event)] {
                    const auto& methodName = event.methodName;
                    auto& methods = dispatchMap();
                    if (methods.contains(methodName)) {
                        auto call = methods.get(methodName);
                        (this->*call)(event);
                    } else
                        LOG_ERROR("Unknown event: %s", methodName.utf8().data());
                });
            }
        });
    });
    result.iterator->value.pushReceivedData(data.span());
}

std::optional<RemoteInspectorConnectionClient::Event> RemoteInspectorConnectionClient::extractEvent(ConnectionID clientID, Vector<uint8_t>&& data)
{
    if (data.isEmpty())
        return std::nullopt;

    String jsonData = String::fromUTF8(data);

    auto messageValue = JSON::Value::parseJSON(jsonData);
    if (!messageValue)
        return std::nullopt;

    auto messageObject = messageValue->asObject();
    if (!messageObject)
        return std::nullopt;

    Event event;

    event.methodName = messageObject->getString("event"_s);
    if (!event.methodName)
        return std::nullopt;

    event.clientID = clientID;

    if (auto connectionID = messageObject->getInteger("connectionID"_s))
        event.connectionID = *connectionID;

    if (auto targetID = messageObject->getInteger("targetID"_s))
        event.targetID = *targetID;

    event.message = messageObject->getString("message"_s);

    return event;
}

std::optional<Vector<Ref<JSON::Object>>> RemoteInspectorConnectionClient::parseTargetListJSON(const String& message)
{
    auto messageValue = JSON::Value::parseJSON(message);
    if (!messageValue)
        return std::nullopt;

    auto messageArray = messageValue->asArray();
    if (!messageArray)
        return std::nullopt;

    Vector<Ref<JSON::Object>> targetList;
    for (auto& itemValue : *messageArray) {
        if (auto itemObject = itemValue->asObject())
            targetList.append(itemObject.releaseNonNull());
        else
            LOG_ERROR("Invalid type of value in targetList. It must be object.");
    }
    return targetList;
}

} // namespace Inspector

#endif // ENABLE(REMOTE_INSPECTOR)

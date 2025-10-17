/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 24, 2024.
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
#include "MessageReceiverMap.h"

#include "Decoder.h"
#include "MessageReceiver.h"

namespace IPC {

MessageReceiverMap::MessageReceiverMap()
{
}

MessageReceiverMap::~MessageReceiverMap()
{
}

void MessageReceiverMap::addMessageReceiver(ReceiverName messageReceiverName, MessageReceiver& messageReceiver)
{
    ASSERT(!m_globalMessageReceivers.contains(messageReceiverName));

    messageReceiver.willBeAddedToMessageReceiverMap();
    m_globalMessageReceivers.set(messageReceiverName, messageReceiver);
}

void MessageReceiverMap::addMessageReceiver(ReceiverName messageReceiverName, uint64_t destinationID, MessageReceiver& messageReceiver)
{
    ASSERT(destinationID);
    ASSERT(!m_messageReceivers.contains(std::make_pair(messageReceiverName, destinationID)));
    ASSERT(!m_globalMessageReceivers.contains(messageReceiverName));

    messageReceiver.willBeAddedToMessageReceiverMap();
    m_messageReceivers.set(std::make_pair(messageReceiverName, destinationID), messageReceiver);
}

void MessageReceiverMap::removeMessageReceiver(ReceiverName messageReceiverName)
{
    auto it = m_globalMessageReceivers.find(messageReceiverName);
    if (it == m_globalMessageReceivers.end()) {
        ASSERT_NOT_REACHED();
        return;
    }

    if (it->value)
        it->value->willBeRemovedFromMessageReceiverMap();
    m_globalMessageReceivers.remove(it);
}

void MessageReceiverMap::removeMessageReceiver(ReceiverName messageReceiverName, uint64_t destinationID)
{
    auto it = m_messageReceivers.find(std::make_pair(messageReceiverName, destinationID));
    if (it == m_messageReceivers.end()) {
        ASSERT_NOT_REACHED();
        return;
    }

    if (it->value)
        it->value->willBeRemovedFromMessageReceiverMap();
    m_messageReceivers.remove(it);
}

void MessageReceiverMap::removeMessageReceiver(MessageReceiver& messageReceiver)
{
    Vector<ReceiverName> globalReceiversToRemove;
    for (auto& [name, receiver] : m_globalMessageReceivers) {
        if (receiver == &messageReceiver)
            globalReceiversToRemove.append(name);
    }

    for (auto& globalReceiverToRemove : globalReceiversToRemove)
        removeMessageReceiver(globalReceiverToRemove);

    Vector<std::pair<ReceiverName, uint64_t>> receiversToRemove;
    for (auto& [nameAndDestinationID, receiver] : m_messageReceivers) {
        if (receiver == &messageReceiver)
            receiversToRemove.append(std::make_pair(nameAndDestinationID.first, nameAndDestinationID.second));
    }

    for (auto& [name, destinationID] : receiversToRemove)
        removeMessageReceiver(name, destinationID);
}

void MessageReceiverMap::invalidate()
{
    for (auto& messageReceiver : m_globalMessageReceivers.values())
        messageReceiver->willBeRemovedFromMessageReceiverMap();
    m_globalMessageReceivers.clear();


    for (auto& messageReceiver : m_messageReceivers.values())
        messageReceiver->willBeRemovedFromMessageReceiverMap();
    m_messageReceivers.clear();
}

bool MessageReceiverMap::dispatchMessage(Connection& connection, Decoder& decoder)
{
    if (auto messageReceiver = m_globalMessageReceivers.get(decoder.messageReceiverName())) {
        ASSERT(!decoder.destinationID());

        messageReceiver->didReceiveMessage(connection, decoder);
        return true;
    }

    if (auto messageReceiver = m_messageReceivers.get(std::make_pair(decoder.messageReceiverName(), decoder.destinationID()))) {
        messageReceiver->didReceiveMessage(connection, decoder);
        return true;
    }

    return false;
}

bool MessageReceiverMap::dispatchSyncMessage(Connection& connection, Decoder& decoder, UniqueRef<Encoder>& replyEncoder)
{
    if (auto messageReceiver = m_globalMessageReceivers.get(decoder.messageReceiverName())) {
        ASSERT(!decoder.destinationID());
        return messageReceiver->didReceiveSyncMessage(connection, decoder, replyEncoder);
    }

    if (auto messageReceiver = m_messageReceivers.get(std::make_pair(decoder.messageReceiverName(), decoder.destinationID())))
        return messageReceiver->didReceiveSyncMessage(connection, decoder, replyEncoder);

    return false;
}

} // namespace IPC

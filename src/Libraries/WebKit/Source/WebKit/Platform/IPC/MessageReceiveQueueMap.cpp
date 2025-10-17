/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 24, 2024.
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
#include "MessageReceiveQueueMap.h"
#include <wtf/text/TextStream.h>

namespace IPC {

void MessageReceiveQueueMap::addImpl(StoreType&& queue, const ReceiverMatcher& matcher)
{
    if (!matcher.receiverName) {
        ASSERT(!m_anyReceiverQueue);
        m_anyReceiverQueue = WTFMove(queue);
        return;
    }
    uint8_t receiverName = static_cast<uint8_t>(*matcher.receiverName);
    if (!matcher.destinationID.has_value()) {
        auto result = m_anyIDQueues.add(receiverName, WTFMove(queue));
        ASSERT_UNUSED(result, result.isNewEntry);
        return;
    }
    auto result = m_queues.add(std::make_pair(receiverName, *matcher.destinationID), WTFMove(queue));
    ASSERT_UNUSED(result, result.isNewEntry);
}

void MessageReceiveQueueMap::remove(const ReceiverMatcher& matcher)
{
    if (!matcher.receiverName) {
        ASSERT(m_anyReceiverQueue);
        m_anyReceiverQueue = nullptr;
        return;
    }
    uint8_t receiverName = static_cast<uint8_t>(*matcher.receiverName);
    if (!matcher.destinationID.has_value()) {
        auto result = m_anyIDQueues.remove(receiverName);
        ASSERT_UNUSED(result, result);
        return;
    }
    auto result = m_queues.remove(std::make_pair(receiverName, *matcher.destinationID));
    ASSERT_UNUSED(result, result);
}

MessageReceiveQueue* MessageReceiveQueueMap::get(const Decoder& message) const
{
    uint8_t receiverName = static_cast<uint8_t>(message.messageReceiverName());
    auto queueExtractor = WTF::makeVisitor(
        [](const std::unique_ptr<MessageReceiveQueue>& queue) { return queue.get(); },
        [](MessageReceiveQueue* queue) { return queue; }
    );

    {
        auto it = m_anyIDQueues.find(receiverName);
        if (it != m_anyIDQueues.end())
            return std::visit(queueExtractor, it->value);
    }
    {
        auto it = m_queues.find(std::make_pair(receiverName, message.destinationID()));
        if (it != m_queues.end())
            return std::visit(queueExtractor, it->value);
    }
    if (m_anyReceiverQueue)
        return std::visit(queueExtractor, *m_anyReceiverQueue);
    return nullptr;
}

}

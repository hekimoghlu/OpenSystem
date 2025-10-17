/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 3, 2025.
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

#include "Decoder.h"
#include "MessageReceiveQueue.h"
#include <variant>
#include <wtf/HashMap.h>

namespace IPC {

enum class ReceiverName : uint8_t;

class MessageReceiveQueueMap {
public:
    MessageReceiveQueueMap() = default;
    ~MessageReceiveQueueMap() = default;

    static bool isValidMessage(const Decoder& message)
    {
        return QueueMap::isValidKey(std::make_pair(static_cast<uint8_t>(message.messageReceiverName()), message.destinationID()));
    }

    void add(MessageReceiveQueue& queue, const ReceiverMatcher& matcher) { addImpl(StoreType(&queue), matcher); }
    void add(std::unique_ptr<MessageReceiveQueue>&& queue, const ReceiverMatcher& matcher) { addImpl(StoreType(WTFMove(queue)), matcher); }
    void remove(const ReceiverMatcher&);

    MessageReceiveQueue* get(const Decoder&) const;
private:
    using StoreType = std::variant<MessageReceiveQueue*, std::unique_ptr<MessageReceiveQueue>>;
    void addImpl(StoreType&&, const ReceiverMatcher&);
    using QueueMap = HashMap<std::pair<uint8_t, uint64_t>, StoreType>;
    // Key is ReceiverName. FIXME: make it possible to use ReceiverName.
    using AnyIDQueueMap = HashMap<uint8_t, StoreType>;
    QueueMap m_queues;
    AnyIDQueueMap m_anyIDQueues;
    std::optional<StoreType> m_anyReceiverQueue;
};

} // namespace IPC

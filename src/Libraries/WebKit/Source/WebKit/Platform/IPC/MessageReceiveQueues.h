/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 9, 2025.
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

#include "Connection.h"
#include "MessageReceiveQueue.h"
#include "WorkQueueMessageReceiver.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/WeakRef.h>

namespace IPC {

class FunctionDispatcherQueue final : public MessageReceiveQueue {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(FunctionDispatcherQueue);
public:
    FunctionDispatcherQueue(FunctionDispatcher& dispatcher, MessageReceiver& receiver)
        : m_dispatcher(dispatcher)
        , m_receiver(receiver)
    {
    }
    ~FunctionDispatcherQueue() final = default;

    void enqueueMessage(Connection& connection, UniqueRef<Decoder>&& message) final
    {
        m_dispatcher.dispatch([connection = Ref { connection }, message = WTFMove(message), receiver = Ref { m_receiver.get() }]() mutable {
            connection->dispatchMessageReceiverMessage(receiver, WTFMove(message));
        });
    }
private:
    FunctionDispatcher& m_dispatcher;
    WeakRef<MessageReceiver> m_receiver;
};

class WorkQueueMessageReceiverQueue final : public MessageReceiveQueue {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(WorkQueueMessageReceiverQueue);
public:
    WorkQueueMessageReceiverQueue(WorkQueue& queue, WorkQueueMessageReceiver& receiver)
        : m_queue(queue)
        , m_receiver(receiver)
    {
    }
    ~WorkQueueMessageReceiverQueue() final = default;

    void enqueueMessage(Connection& connection, UniqueRef<Decoder>&& message) final
    {
        Ref { m_queue }->dispatch([connection = Ref { connection }, message = WTFMove(message), receiver = m_receiver]() mutable {
            connection->dispatchMessageReceiverMessage(receiver.get(), WTFMove(message));
        });
    }
private:
    Ref<WorkQueue> m_queue;
    Ref<WorkQueueMessageReceiver> m_receiver;
};

} // namespace IPC

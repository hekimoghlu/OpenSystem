/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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

#include "MessagePortChannelProvider.h"
#include <wtf/CompletionHandler.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class WorkerOrWorkletGlobalScope;

class WorkerMessagePortChannelProvider final : public MessagePortChannelProvider, public CanMakeCheckedPtr<WorkerMessagePortChannelProvider> {
    WTF_MAKE_TZONE_ALLOCATED(WorkerMessagePortChannelProvider);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WorkerMessagePortChannelProvider);
public:
    explicit WorkerMessagePortChannelProvider(WorkerOrWorkletGlobalScope&);
    ~WorkerMessagePortChannelProvider();

private:
    void createNewMessagePortChannel(const MessagePortIdentifier& local, const MessagePortIdentifier& remote) final;
    void entangleLocalPortInThisProcessToRemote(const MessagePortIdentifier& local, const MessagePortIdentifier& remote) final;
    void messagePortDisentangled(const MessagePortIdentifier& local) final;
    void messagePortClosed(const MessagePortIdentifier& local) final;
    void postMessageToRemote(MessageWithMessagePorts&&, const MessagePortIdentifier& remoteTarget) final;
    void takeAllMessagesForPort(const MessagePortIdentifier&, CompletionHandler<void(Vector<MessageWithMessagePorts>&&, CompletionHandler<void()>&&)>&&) final;

    WeakRef<WorkerOrWorkletGlobalScope> m_scope;

    uint64_t m_lastCallbackIdentifier { 0 };
    UncheckedKeyHashMap<uint64_t, CompletionHandler<void(Vector<MessageWithMessagePorts>&&, Function<void()>&&)>> m_takeAllMessagesCallbacks;
};

} // namespace WebCore

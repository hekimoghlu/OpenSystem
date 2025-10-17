/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 13, 2024.
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
#include "MessageChannel.h"

#include "MessagePort.h"
#include "MessagePortChannelProvider.h"
#include "ScriptExecutionContext.h"

namespace WebCore {

static std::pair<Ref<MessagePort>, Ref<MessagePort>> generateMessagePorts(ScriptExecutionContext& context)
{
    MessagePortIdentifier id1 = { Process::identifier(), PortIdentifier::generate() };
    MessagePortIdentifier id2 = { Process::identifier(), PortIdentifier::generate() };

    return { MessagePort::create(context, id1, id2), MessagePort::create(context, id2, id1) };
}

Ref<MessageChannel> MessageChannel::create(ScriptExecutionContext& context)
{
    return adoptRef(*new MessageChannel(context));
}

MessageChannel::MessageChannel(ScriptExecutionContext& context)
    : m_ports(generateMessagePorts(context))
{
    if (!context.activeDOMObjectsAreStopped()) {
        ASSERT(!port1().isDetached());
        ASSERT(!port2().isDetached());
        MessagePortChannelProvider::fromContext(context).createNewMessagePortChannel(port1().identifier(), port2().identifier());
    } else {
        ASSERT(port1().isDetached());
        ASSERT(port2().isDetached());
    }
}

MessageChannel::~MessageChannel() = default;

} // namespace WebCore

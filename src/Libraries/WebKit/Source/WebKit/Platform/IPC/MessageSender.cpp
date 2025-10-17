/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 5, 2023.
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
#include "MessageSender.h"

#include "Connection.h"

namespace IPC {

MessageSender::~MessageSender() = default;

bool MessageSender::sendMessage(UniqueRef<Encoder>&& encoder, OptionSet<SendOption> sendOptions)
{
    RefPtr connection = messageSenderConnection();
    ASSERT(connection);
    // FIXME: Propagate errors out.
    return connection->sendMessage(WTFMove(encoder), sendOptions) == Error::NoError;
}

bool MessageSender::sendMessageWithAsyncReply(UniqueRef<Encoder>&& encoder, AsyncReplyHandler replyHandler, OptionSet<SendOption> sendOptions)
{
    RefPtr connection = messageSenderConnection();
    ASSERT(connection);
    // FIXME: Propagate errors out.
    return connection->sendMessageWithAsyncReply(WTFMove(encoder), WTFMove(replyHandler), sendOptions) == Error::NoError;
}

bool MessageSender::performSendWithoutUsingIPCConnection(UniqueRef<Encoder>&&) const
{
    // Senders that use sendWithoutUsingIPCConnection(T&& message) must also override this.
    RELEASE_ASSERT_NOT_REACHED();
}

bool MessageSender::performSendWithAsyncReplyWithoutUsingIPCConnection(UniqueRef<Encoder>&&, CompletionHandler<void(Decoder*)>&&) const
{
    // Senders that use sendWithAsyncReplyWithoutUsingIPCConnection(T&& message, C&& completionHandler) must also override this.
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace IPC

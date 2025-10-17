/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
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
#import "config.h"
#import "WebPushDaemonConnection.h"

#if PLATFORM(COCOA) && ENABLE(WEB_PUSH_NOTIFICATIONS)

#import "DaemonUtilities.h"
#import "Decoder.h"
#import "HandleMessage.h"
#import "MessageSenderInlines.h"
#import "PushClientConnectionMessages.h"
#import "WebPushDaemonConstants.h"
#import <wtf/spi/darwin/XPCSPI.h>

namespace WebKit::WebPushD { 

void Connection::newConnectionWasInitialized() const
{
    ASSERT(m_connection);
    sendWithoutUsingIPCConnection(Messages::PushClientConnection::InitializeConnection(m_configuration));
}

static OSObjectPtr<xpc_object_t> messageDictionaryFromEncoder(UniqueRef<IPC::Encoder>&& encoder)
{
    auto xpcData = encoderToXPCData(WTFMove(encoder));
    auto dictionary = adoptOSObject(xpc_dictionary_create(nullptr, nullptr, 0));
    xpc_dictionary_set_uint64(dictionary.get(), WebPushD::protocolVersionKey, WebPushD::protocolVersionValue);
    xpc_dictionary_set_value(dictionary.get(), WebPushD::protocolEncodedMessageKey, xpcData.get());

    return dictionary;
}

bool Connection::performSendWithoutUsingIPCConnection(UniqueRef<IPC::Encoder>&& encoder) const
{
    auto dictionary = messageDictionaryFromEncoder(WTFMove(encoder));
    Daemon::Connection::send(dictionary.get());

    return true;
}

bool Connection::performSendWithAsyncReplyWithoutUsingIPCConnection(UniqueRef<IPC::Encoder>&& encoder, CompletionHandler<void(IPC::Decoder*)>&& completionHandler) const
{
    auto dictionary = messageDictionaryFromEncoder(WTFMove(encoder));
    Daemon::Connection::sendWithReply(dictionary.get(), [completionHandler = WTFMove(completionHandler)] (xpc_object_t reply) mutable {
        if (xpc_get_type(reply) != XPC_TYPE_DICTIONARY)
            return completionHandler(nullptr);

        if (xpc_dictionary_get_uint64(reply, WebPushD::protocolVersionKey) != WebPushD::protocolVersionValue)
            return completionHandler(nullptr);

        auto data = xpc_dictionary_get_data_span(reply, WebPushD::protocolEncodedMessageKey);
        auto decoder = IPC::Decoder::create(data, { });

        completionHandler(decoder.get());
    });

    return true;
}

} // namespace WebKit::WebPushD

#endif // PLATFORM(COCOA) && ENABLE(WEB_PUSH_NOTIFICATIONS)

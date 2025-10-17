/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 7, 2023.
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
#include "TestWithoutUsingIPCConnection.h"

#include "ArgumentCoders.h" // NOLINT
#include "Decoder.h" // NOLINT
#include "HandleMessage.h" // NOLINT
#include "TestWithoutUsingIPCConnectionMessages.h" // NOLINT
#include <wtf/text/WTFString.h> // NOLINT

#if ENABLE(IPC_TESTING_API)
#include "JSIPCBinding.h"
#endif

namespace WebKit {

void TestWithoutUsingIPCConnection::didReceiveMessageWithReplyHandler(IPC::Decoder& decoder, Function<void(UniqueRef<IPC::Encoder>&&)>&& replyHandler)
{
    Ref protectedThis { *this };
    if (decoder.messageName() == Messages::TestWithoutUsingIPCConnection::MessageWithoutArgument::name())
        return IPC::handleMessageWithoutUsingIPCConnection<Messages::TestWithoutUsingIPCConnection::MessageWithoutArgument>(decoder, this, &TestWithoutUsingIPCConnection::messageWithoutArgument);
    if (decoder.messageName() == Messages::TestWithoutUsingIPCConnection::MessageWithoutArgumentAndEmptyReply::name())
        return IPC::handleMessageAsyncWithoutUsingIPCConnection<Messages::TestWithoutUsingIPCConnection::MessageWithoutArgumentAndEmptyReply>(decoder, WTFMove(replyHandler), this, &TestWithoutUsingIPCConnection::messageWithoutArgumentAndEmptyReply);
    if (decoder.messageName() == Messages::TestWithoutUsingIPCConnection::MessageWithoutArgumentAndReplyWithArgument::name())
        return IPC::handleMessageAsyncWithoutUsingIPCConnection<Messages::TestWithoutUsingIPCConnection::MessageWithoutArgumentAndReplyWithArgument>(decoder, WTFMove(replyHandler), this, &TestWithoutUsingIPCConnection::messageWithoutArgumentAndReplyWithArgument);
    if (decoder.messageName() == Messages::TestWithoutUsingIPCConnection::MessageWithArgument::name())
        return IPC::handleMessageWithoutUsingIPCConnection<Messages::TestWithoutUsingIPCConnection::MessageWithArgument>(decoder, this, &TestWithoutUsingIPCConnection::messageWithArgument);
    if (decoder.messageName() == Messages::TestWithoutUsingIPCConnection::MessageWithArgumentAndEmptyReply::name())
        return IPC::handleMessageAsyncWithoutUsingIPCConnection<Messages::TestWithoutUsingIPCConnection::MessageWithArgumentAndEmptyReply>(decoder, WTFMove(replyHandler), this, &TestWithoutUsingIPCConnection::messageWithArgumentAndEmptyReply);
    if (decoder.messageName() == Messages::TestWithoutUsingIPCConnection::MessageWithArgumentAndReplyWithArgument::name())
        return IPC::handleMessageAsyncWithoutUsingIPCConnection<Messages::TestWithoutUsingIPCConnection::MessageWithArgumentAndReplyWithArgument>(decoder, WTFMove(replyHandler), this, &TestWithoutUsingIPCConnection::messageWithArgumentAndReplyWithArgument);
    RELEASE_LOG_ERROR(IPC, "Unhandled message %s to %" PRIu64, IPC::description(decoder.messageName()).characters(), decoder.destinationID());
    decoder.markInvalid();
}

} // namespace WebKit

#if ENABLE(IPC_TESTING_API)

namespace IPC {

template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithoutUsingIPCConnection_MessageWithoutArgument>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithoutUsingIPCConnection::MessageWithoutArgument::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithoutUsingIPCConnection_MessageWithoutArgumentAndEmptyReply>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithoutUsingIPCConnection::MessageWithoutArgumentAndEmptyReply::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessageReply<MessageName::TestWithoutUsingIPCConnection_MessageWithoutArgumentAndEmptyReply>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithoutUsingIPCConnection::MessageWithoutArgumentAndEmptyReply::ReplyArguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithoutUsingIPCConnection_MessageWithoutArgumentAndReplyWithArgument>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithoutUsingIPCConnection::MessageWithoutArgumentAndReplyWithArgument::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessageReply<MessageName::TestWithoutUsingIPCConnection_MessageWithoutArgumentAndReplyWithArgument>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithoutUsingIPCConnection::MessageWithoutArgumentAndReplyWithArgument::ReplyArguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithoutUsingIPCConnection_MessageWithArgument>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithoutUsingIPCConnection::MessageWithArgument::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithoutUsingIPCConnection_MessageWithArgumentAndEmptyReply>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithoutUsingIPCConnection::MessageWithArgumentAndEmptyReply::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessageReply<MessageName::TestWithoutUsingIPCConnection_MessageWithArgumentAndEmptyReply>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithoutUsingIPCConnection::MessageWithArgumentAndEmptyReply::ReplyArguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithoutUsingIPCConnection_MessageWithArgumentAndReplyWithArgument>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithoutUsingIPCConnection::MessageWithArgumentAndReplyWithArgument::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessageReply<MessageName::TestWithoutUsingIPCConnection_MessageWithArgumentAndReplyWithArgument>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithoutUsingIPCConnection::MessageWithArgumentAndReplyWithArgument::ReplyArguments>(globalObject, decoder);
}

}

#endif


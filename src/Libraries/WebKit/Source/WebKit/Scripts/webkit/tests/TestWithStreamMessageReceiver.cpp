/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 16, 2022.
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
#include "TestWithStream.h"

#include "ArgumentCoders.h" // NOLINT
#include "Decoder.h" // NOLINT
#include "HandleMessage.h" // NOLINT
#include "TestWithStreamMessages.h" // NOLINT
#if PLATFORM(COCOA)
#include <wtf/MachSendRight.h> // NOLINT
#endif
#include <wtf/text/WTFString.h> // NOLINT

#if ENABLE(IPC_TESTING_API)
#include "JSIPCBinding.h"
#endif

namespace WebKit {

void TestWithStream::didReceiveStreamMessage(IPC::StreamServerConnection& connection, IPC::Decoder& decoder)
{
    Ref protectedThis { *this };
    if (decoder.messageName() == Messages::TestWithStream::SendString::name())
        return IPC::handleMessage<Messages::TestWithStream::SendString>(connection, decoder, this, &TestWithStream::sendString);
    if (decoder.messageName() == Messages::TestWithStream::SendStringAsync::name())
        return IPC::handleMessageAsync<Messages::TestWithStream::SendStringAsync>(connection, decoder, this, &TestWithStream::sendStringAsync);
    if (decoder.messageName() == Messages::TestWithStream::CallWithIdentifier::name())
        return IPC::handleMessageAsyncWithReplyID<Messages::TestWithStream::CallWithIdentifier>(connection, decoder, this, &TestWithStream::callWithIdentifier);
#if PLATFORM(COCOA)
    if (decoder.messageName() == Messages::TestWithStream::SendMachSendRight::name())
        return IPC::handleMessage<Messages::TestWithStream::SendMachSendRight>(connection, decoder, this, &TestWithStream::sendMachSendRight);
#endif
    if (decoder.messageName() == Messages::TestWithStream::SendStringSync::name())
        return IPC::handleMessageSynchronous<Messages::TestWithStream::SendStringSync>(connection, decoder, this, &TestWithStream::sendStringSync);
#if PLATFORM(COCOA)
    if (decoder.messageName() == Messages::TestWithStream::ReceiveMachSendRight::name())
        return IPC::handleMessageSynchronous<Messages::TestWithStream::ReceiveMachSendRight>(connection, decoder, this, &TestWithStream::receiveMachSendRight);
    if (decoder.messageName() == Messages::TestWithStream::SendAndReceiveMachSendRight::name())
        return IPC::handleMessageSynchronous<Messages::TestWithStream::SendAndReceiveMachSendRight>(connection, decoder, this, &TestWithStream::sendAndReceiveMachSendRight);
#endif
    RELEASE_LOG_ERROR(IPC, "Unhandled stream message %s to %" PRIu64, IPC::description(decoder.messageName()).characters(), decoder.destinationID());
    decoder.markInvalid();
}

} // namespace WebKit

#if ENABLE(IPC_TESTING_API)

namespace IPC {

template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithStream_SendString>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithStream::SendString::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithStream_SendStringAsync>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithStream::SendStringAsync::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessageReply<MessageName::TestWithStream_SendStringAsync>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithStream::SendStringAsync::ReplyArguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithStream_SendStringSync>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithStream::SendStringSync::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessageReply<MessageName::TestWithStream_SendStringSync>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithStream::SendStringSync::ReplyArguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithStream_CallWithIdentifier>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithStream::CallWithIdentifier::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessageReply<MessageName::TestWithStream_CallWithIdentifier>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithStream::CallWithIdentifier::ReplyArguments>(globalObject, decoder);
}
#if PLATFORM(COCOA)
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithStream_SendMachSendRight>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithStream::SendMachSendRight::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithStream_ReceiveMachSendRight>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithStream::ReceiveMachSendRight::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessageReply<MessageName::TestWithStream_ReceiveMachSendRight>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithStream::ReceiveMachSendRight::ReplyArguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithStream_SendAndReceiveMachSendRight>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithStream::SendAndReceiveMachSendRight::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessageReply<MessageName::TestWithStream_SendAndReceiveMachSendRight>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithStream::SendAndReceiveMachSendRight::ReplyArguments>(globalObject, decoder);
}
#endif

}

#endif


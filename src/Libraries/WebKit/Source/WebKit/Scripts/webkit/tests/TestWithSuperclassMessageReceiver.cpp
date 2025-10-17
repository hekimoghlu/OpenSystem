/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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
#include "TestWithSuperclass.h"

#include "ArgumentCoders.h" // NOLINT
#include "Decoder.h" // NOLINT
#include "HandleMessage.h" // NOLINT
#include "TestClassName.h" // NOLINT
#if ENABLE(TEST_FEATURE)
#include "TestTwoStateEnum.h" // NOLINT
#endif
#include "TestWithSuperclassMessages.h" // NOLINT
#include <optional> // NOLINT
#include <wtf/text/WTFString.h> // NOLINT

#if ENABLE(IPC_TESTING_API)
#include "JSIPCBinding.h"
#endif

namespace WebKit {

void TestWithSuperclass::didReceiveMessage(IPC::Connection& connection, IPC::Decoder& decoder)
{
    Ref protectedThis { *this };
    if (decoder.messageName() == Messages::TestWithSuperclass::LoadURL::name())
        return IPC::handleMessage<Messages::TestWithSuperclass::LoadURL>(connection, decoder, this, &TestWithSuperclass::loadURL);
#if ENABLE(TEST_FEATURE)
    if (decoder.messageName() == Messages::TestWithSuperclass::TestAsyncMessage::name())
        return IPC::handleMessageAsync<Messages::TestWithSuperclass::TestAsyncMessage>(connection, decoder, this, &TestWithSuperclass::testAsyncMessage);
    if (decoder.messageName() == Messages::TestWithSuperclass::TestAsyncMessageWithNoArguments::name())
        return IPC::handleMessageAsync<Messages::TestWithSuperclass::TestAsyncMessageWithNoArguments>(connection, decoder, this, &TestWithSuperclass::testAsyncMessageWithNoArguments);
    if (decoder.messageName() == Messages::TestWithSuperclass::TestAsyncMessageWithMultipleArguments::name())
        return IPC::handleMessageAsync<Messages::TestWithSuperclass::TestAsyncMessageWithMultipleArguments>(connection, decoder, this, &TestWithSuperclass::testAsyncMessageWithMultipleArguments);
    if (decoder.messageName() == Messages::TestWithSuperclass::TestAsyncMessageWithConnection::name())
        return IPC::handleMessageAsync<Messages::TestWithSuperclass::TestAsyncMessageWithConnection>(connection, decoder, this, &TestWithSuperclass::testAsyncMessageWithConnection);
#endif
    WebPageBase::didReceiveMessage(connection, decoder);
}

bool TestWithSuperclass::didReceiveSyncMessage(IPC::Connection& connection, IPC::Decoder& decoder, UniqueRef<IPC::Encoder>& replyEncoder)
{
    Ref protectedThis { *this };
    if (decoder.messageName() == Messages::TestWithSuperclass::TestSyncMessage::name())
        return IPC::handleMessageSynchronous<Messages::TestWithSuperclass::TestSyncMessage>(connection, decoder, replyEncoder, this, &TestWithSuperclass::testSyncMessage);
    if (decoder.messageName() == Messages::TestWithSuperclass::TestSynchronousMessage::name())
        return IPC::handleMessageSynchronous<Messages::TestWithSuperclass::TestSynchronousMessage>(connection, decoder, replyEncoder, this, &TestWithSuperclass::testSynchronousMessage);
    return WebPageBase::didReceiveSyncMessage(connection, decoder, replyEncoder);
}

} // namespace WebKit

#if ENABLE(IPC_TESTING_API)

namespace IPC {

template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithSuperclass_LoadURL>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithSuperclass::LoadURL::Arguments>(globalObject, decoder);
}
#if ENABLE(TEST_FEATURE)
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithSuperclass_TestAsyncMessage>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithSuperclass::TestAsyncMessage::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessageReply<MessageName::TestWithSuperclass_TestAsyncMessage>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithSuperclass::TestAsyncMessage::ReplyArguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithSuperclass_TestAsyncMessageWithNoArguments>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithSuperclass::TestAsyncMessageWithNoArguments::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessageReply<MessageName::TestWithSuperclass_TestAsyncMessageWithNoArguments>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithSuperclass::TestAsyncMessageWithNoArguments::ReplyArguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithSuperclass_TestAsyncMessageWithMultipleArguments>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithSuperclass::TestAsyncMessageWithMultipleArguments::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessageReply<MessageName::TestWithSuperclass_TestAsyncMessageWithMultipleArguments>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithSuperclass::TestAsyncMessageWithMultipleArguments::ReplyArguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithSuperclass_TestAsyncMessageWithConnection>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithSuperclass::TestAsyncMessageWithConnection::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessageReply<MessageName::TestWithSuperclass_TestAsyncMessageWithConnection>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithSuperclass::TestAsyncMessageWithConnection::ReplyArguments>(globalObject, decoder);
}
#endif
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithSuperclass_TestSyncMessage>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithSuperclass::TestSyncMessage::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessageReply<MessageName::TestWithSuperclass_TestSyncMessage>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithSuperclass::TestSyncMessage::ReplyArguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithSuperclass_TestSynchronousMessage>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithSuperclass::TestSynchronousMessage::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessageReply<MessageName::TestWithSuperclass_TestSynchronousMessage>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithSuperclass::TestSynchronousMessage::ReplyArguments>(globalObject, decoder);
}

}

#endif


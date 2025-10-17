/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 13, 2022.
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
#include "TestWithSuperclassAndWantsAsyncDispatch.h"

#include "ArgumentCoders.h" // NOLINT
#include "Decoder.h" // NOLINT
#include "HandleMessage.h" // NOLINT
#include "TestWithSuperclassAndWantsAsyncDispatchMessages.h" // NOLINT
#include <wtf/text/WTFString.h> // NOLINT

#if ENABLE(IPC_TESTING_API)
#include "JSIPCBinding.h"
#endif

namespace WebKit {

void TestWithSuperclassAndWantsAsyncDispatch::didReceiveMessage(IPC::Connection& connection, IPC::Decoder& decoder)
{
    Ref protectedThis { *this };
    if (decoder.messageName() == Messages::TestWithSuperclassAndWantsAsyncDispatch::LoadURL::name())
        return IPC::handleMessage<Messages::TestWithSuperclassAndWantsAsyncDispatch::LoadURL>(connection, decoder, this, &TestWithSuperclassAndWantsAsyncDispatch::loadURL);
    if (dispatchMessage(connection, decoder))
        return;
    WebPageBase::didReceiveMessage(connection, decoder);
}

bool TestWithSuperclassAndWantsAsyncDispatch::didReceiveSyncMessage(IPC::Connection& connection, IPC::Decoder& decoder, UniqueRef<IPC::Encoder>& replyEncoder)
{
    Ref protectedThis { *this };
    if (decoder.messageName() == Messages::TestWithSuperclassAndWantsAsyncDispatch::TestSyncMessage::name())
        return IPC::handleMessageSynchronous<Messages::TestWithSuperclassAndWantsAsyncDispatch::TestSyncMessage>(connection, decoder, replyEncoder, this, &TestWithSuperclassAndWantsAsyncDispatch::testSyncMessage);
    return WebPageBase::didReceiveSyncMessage(connection, decoder, replyEncoder);
}

} // namespace WebKit

#if ENABLE(IPC_TESTING_API)

namespace IPC {

template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithSuperclassAndWantsAsyncDispatch_LoadURL>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithSuperclassAndWantsAsyncDispatch::LoadURL::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithSuperclassAndWantsAsyncDispatch_TestSyncMessage>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithSuperclassAndWantsAsyncDispatch::TestSyncMessage::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessageReply<MessageName::TestWithSuperclassAndWantsAsyncDispatch_TestSyncMessage>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithSuperclassAndWantsAsyncDispatch::TestSyncMessage::ReplyArguments>(globalObject, decoder);
}

}

#endif


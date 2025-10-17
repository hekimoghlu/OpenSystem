/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 16, 2023.
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
#include "TestWithSuperclassAndWantsDispatch.h"

#include "ArgumentCoders.h" // NOLINT
#include "Decoder.h" // NOLINT
#include "HandleMessage.h" // NOLINT
#include "TestWithSuperclassAndWantsDispatchMessages.h" // NOLINT
#include <wtf/text/WTFString.h> // NOLINT

#if ENABLE(IPC_TESTING_API)
#include "JSIPCBinding.h"
#endif

namespace WebKit {

void TestWithSuperclassAndWantsDispatch::didReceiveMessage(IPC::Connection& connection, IPC::Decoder& decoder)
{
    Ref protectedThis { *this };
    if (decoder.messageName() == Messages::TestWithSuperclassAndWantsDispatch::LoadURL::name())
        return IPC::handleMessage<Messages::TestWithSuperclassAndWantsDispatch::LoadURL>(connection, decoder, this, &TestWithSuperclassAndWantsDispatch::loadURL);
    if (dispatchMessage(connection, decoder))
        return;
    WebPageBase::didReceiveMessage(connection, decoder);
}

bool TestWithSuperclassAndWantsDispatch::didReceiveSyncMessage(IPC::Connection& connection, IPC::Decoder& decoder, UniqueRef<IPC::Encoder>& replyEncoder)
{
    Ref protectedThis { *this };
    if (decoder.messageName() == Messages::TestWithSuperclassAndWantsDispatch::TestSyncMessage::name())
        return IPC::handleMessageSynchronous<Messages::TestWithSuperclassAndWantsDispatch::TestSyncMessage>(connection, decoder, replyEncoder, this, &TestWithSuperclassAndWantsDispatch::testSyncMessage);
    if (dispatchSyncMessage(connection, decoder, replyEncoder))
        return true;
    return WebPageBase::didReceiveSyncMessage(connection, decoder, replyEncoder);
}

} // namespace WebKit

#if ENABLE(IPC_TESTING_API)

namespace IPC {

template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithSuperclassAndWantsDispatch_LoadURL>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithSuperclassAndWantsDispatch::LoadURL::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithSuperclassAndWantsDispatch_TestSyncMessage>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithSuperclassAndWantsDispatch::TestSyncMessage::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessageReply<MessageName::TestWithSuperclassAndWantsDispatch_TestSyncMessage>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithSuperclassAndWantsDispatch::TestSyncMessage::ReplyArguments>(globalObject, decoder);
}

}

#endif


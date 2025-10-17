/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 17, 2024.
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
#if ENABLE(APPLE_PAY)
#include "TestWithIfReceiver.h"

#include "Decoder.h"
#include "HandleMessage.h"
#include "TestWithIfReceiverMessages.h"
#include <WebCore/ApplePayPaymentAuthorizationResult.h>

#if ENABLE(IPC_TESTING_API)
#include "JSIPCBinding.h"
#endif

namespace WebKit {

void TestWithIfReceiver::didReceiveMessage(IPC::Connection& connection, IPC::Decoder& decoder)
{
    if (decoder.messageName() == Messages::TestWithIfReceiver::CompletePaymentSession::name())
        return IPC::handleMessage<Messages::TestWithIfReceiver::CompletePaymentSession>(connection, decoder, this, &TestWithIfReceiver::completePaymentSession);
    UNUSED_PARAM(connection);
    UNUSED_PARAM(decoder);
#if ENABLE(IPC_TESTING_API)
    if (connection.ignoreInvalidMessageForTesting())
        return;
#endif // ENABLE(IPC_TESTING_API)
    ASSERT_NOT_REACHED_WITH_MESSAGE("Unhandled message %s to %" PRIu64, IPC::description(decoder.messageName()), decoder.destinationID());
}

} // namespace WebKit

#if ENABLE(IPC_TESTING_API)

namespace IPC {

template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithIfReceiver_CompletePaymentSession>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithIfReceiver::CompletePaymentSession::Arguments>(globalObject, decoder);
}

}

#endif


#endif // ENABLE(APPLE_PAY)

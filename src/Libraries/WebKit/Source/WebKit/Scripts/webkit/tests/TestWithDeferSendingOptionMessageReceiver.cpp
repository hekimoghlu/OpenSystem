/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 22, 2024.
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
#include "TestWithDeferSendingOption.h"

#include "ArgumentCoders.h" // NOLINT
#include "Decoder.h" // NOLINT
#include "HandleMessage.h" // NOLINT
#include "SharedPreferencesForWebProcess.h" // NOLINT
#include "TestWithDeferSendingOptionMessages.h" // NOLINT
#include <wtf/text/WTFString.h> // NOLINT

#if ENABLE(IPC_TESTING_API)
#include "JSIPCBinding.h"
#endif

namespace WebKit {

void TestWithDeferSendingOption::didReceiveMessage(IPC::Connection& connection, IPC::Decoder& decoder)
{
    auto sharedPreferences = sharedPreferencesForWebProcess();
    UNUSED_VARIABLE(sharedPreferences);
    if (!sharedPreferences || !sharedPreferences->someFeature) {
        RELEASE_LOG_ERROR(IPC, "Message %s received by a disabled message receiver TestWithDeferSendingOption", IPC::description(decoder.messageName()).characters());
        decoder.markInvalid();
        return;
    }
    Ref protectedThis { *this };
    if (decoder.messageName() == Messages::TestWithDeferSendingOption::NoOptions::name())
        return IPC::handleMessage<Messages::TestWithDeferSendingOption::NoOptions>(connection, decoder, this, &TestWithDeferSendingOption::noOptions);
    if (decoder.messageName() == Messages::TestWithDeferSendingOption::NoIndices::name())
        return IPC::handleMessage<Messages::TestWithDeferSendingOption::NoIndices>(connection, decoder, this, &TestWithDeferSendingOption::noIndices);
    if (decoder.messageName() == Messages::TestWithDeferSendingOption::OneIndex::name())
        return IPC::handleMessage<Messages::TestWithDeferSendingOption::OneIndex>(connection, decoder, this, &TestWithDeferSendingOption::oneIndex);
    if (decoder.messageName() == Messages::TestWithDeferSendingOption::MultipleIndices::name())
        return IPC::handleMessage<Messages::TestWithDeferSendingOption::MultipleIndices>(connection, decoder, this, &TestWithDeferSendingOption::multipleIndices);
    UNUSED_PARAM(connection);
    RELEASE_LOG_ERROR(IPC, "Unhandled message %s to %" PRIu64, IPC::description(decoder.messageName()).characters(), decoder.destinationID());
    decoder.markInvalid();
}

} // namespace WebKit

#if ENABLE(IPC_TESTING_API)

namespace IPC {

template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithDeferSendingOption_NoOptions>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithDeferSendingOption::NoOptions::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithDeferSendingOption_NoIndices>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithDeferSendingOption::NoIndices::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithDeferSendingOption_OneIndex>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithDeferSendingOption::OneIndex::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithDeferSendingOption_MultipleIndices>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithDeferSendingOption::MultipleIndices::Arguments>(globalObject, decoder);
}

}

#endif


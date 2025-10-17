/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 24, 2024.
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
#include "TestWithEnabledBy.h"

#include "ArgumentCoders.h" // NOLINT
#include "Decoder.h" // NOLINT
#include "HandleMessage.h" // NOLINT
#include "SharedPreferencesForWebProcess.h" // NOLINT
#include "TestWithEnabledByMessages.h" // NOLINT
#include <wtf/text/WTFString.h> // NOLINT

#if ENABLE(IPC_TESTING_API)
#include "JSIPCBinding.h"
#endif

namespace WebKit {

void TestWithEnabledBy::didReceiveMessage(IPC::Connection& connection, IPC::Decoder& decoder)
{
    auto sharedPreferences = sharedPreferencesForWebProcess(connection);
    UNUSED_VARIABLE(sharedPreferences);
    Ref protectedThis { *this };
    if (decoder.messageName() == Messages::TestWithEnabledBy::AlwaysEnabled::name())
        return IPC::handleMessage<Messages::TestWithEnabledBy::AlwaysEnabled>(connection, decoder, this, &TestWithEnabledBy::alwaysEnabled);
    if (decoder.messageName() == Messages::TestWithEnabledBy::ConditionallyEnabled::name()) {
        if (!(sharedPreferences && sharedPreferences->someFeature)) {
            RELEASE_LOG_ERROR(IPC, "Message %s received by a disabled message endpoint", IPC::description(decoder.messageName()).characters());
            return decoder.markInvalid();
        }
        return IPC::handleMessage<Messages::TestWithEnabledBy::ConditionallyEnabled>(connection, decoder, this, &TestWithEnabledBy::conditionallyEnabled);
    }
    if (decoder.messageName() == Messages::TestWithEnabledBy::ConditionallyEnabledAnd::name()) {
        if (!(sharedPreferences && (sharedPreferences->someFeature && sharedPreferences->otherFeature))) {
            RELEASE_LOG_ERROR(IPC, "Message %s received by a disabled message endpoint", IPC::description(decoder.messageName()).characters());
            return decoder.markInvalid();
        }
        return IPC::handleMessage<Messages::TestWithEnabledBy::ConditionallyEnabledAnd>(connection, decoder, this, &TestWithEnabledBy::conditionallyEnabledAnd);
    }
    if (decoder.messageName() == Messages::TestWithEnabledBy::ConditionallyEnabledOr::name()) {
        if (!(sharedPreferences && (sharedPreferences->someFeature || sharedPreferences->otherFeature))) {
            RELEASE_LOG_ERROR(IPC, "Message %s received by a disabled message endpoint", IPC::description(decoder.messageName()).characters());
            return decoder.markInvalid();
        }
        return IPC::handleMessage<Messages::TestWithEnabledBy::ConditionallyEnabledOr>(connection, decoder, this, &TestWithEnabledBy::conditionallyEnabledOr);
    }
    UNUSED_PARAM(connection);
    RELEASE_LOG_ERROR(IPC, "Unhandled message %s to %" PRIu64, IPC::description(decoder.messageName()).characters(), decoder.destinationID());
    decoder.markInvalid();
}

} // namespace WebKit

#if ENABLE(IPC_TESTING_API)

namespace IPC {

template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithEnabledBy_AlwaysEnabled>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithEnabledBy::AlwaysEnabled::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithEnabledBy_ConditionallyEnabled>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithEnabledBy::ConditionallyEnabled::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithEnabledBy_ConditionallyEnabledAnd>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithEnabledBy::ConditionallyEnabledAnd::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithEnabledBy_ConditionallyEnabledOr>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithEnabledBy::ConditionallyEnabledOr::Arguments>(globalObject, decoder);
}

}

#endif


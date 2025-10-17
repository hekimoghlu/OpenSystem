/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 7, 2023.
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
#include "TestWithValidator.h"

#include "ArgumentCoders.h" // NOLINT
#include "Decoder.h" // NOLINT
#include "HandleMessage.h" // NOLINT
#include "SharedPreferencesForWebProcess.h" // NOLINT
#include "TestWithValidatorMessages.h" // NOLINT
#include <wtf/text/WTFString.h> // NOLINT

#if ENABLE(IPC_TESTING_API)
#include "JSIPCBinding.h"
#endif

namespace WebKit {

void TestWithValidator::didReceiveMessage(IPC::Connection& connection, IPC::Decoder& decoder)
{
    auto sharedPreferences = sharedPreferencesForWebProcess();
    UNUSED_VARIABLE(sharedPreferences);
    if (!sharedPreferences || !sharedPreferences->someOtherFeature) {
        RELEASE_LOG_ERROR(IPC, "Message %s received by a disabled message receiver TestWithValidator", IPC::description(decoder.messageName()).characters());
        decoder.markInvalid();
        return;
    }
    Ref protectedThis { *this };
    if (decoder.messageName() == Messages::TestWithValidator::AlwaysEnabled::name())
        return IPC::handleMessage<Messages::TestWithValidator::AlwaysEnabled>(connection, decoder, this, &TestWithValidator::alwaysEnabled);
    if (decoder.messageName() == Messages::TestWithValidator::EnabledIfPassValidation::name()) {
        if (!ValidateFunction()) {
            RELEASE_LOG_ERROR(IPC, "Message %s fails validation", IPC::description(decoder.messageName()).characters());
            return decoder.markInvalid();
        }
        return IPC::handleMessage<Messages::TestWithValidator::EnabledIfPassValidation>(connection, decoder, this, &TestWithValidator::enabledIfPassValidation);
    }
    if (decoder.messageName() == Messages::TestWithValidator::EnabledIfSomeFeatureEnabledAndPassValidation::name()) {
        if (!(sharedPreferences && sharedPreferences->someFeature)) {
            RELEASE_LOG_ERROR(IPC, "Message %s received by a disabled message endpoint", IPC::description(decoder.messageName()).characters());
            return decoder.markInvalid();
        }
        if (!ValidateFunction()) {
            RELEASE_LOG_ERROR(IPC, "Message %s fails validation", IPC::description(decoder.messageName()).characters());
            return decoder.markInvalid();
        }
        return IPC::handleMessage<Messages::TestWithValidator::EnabledIfSomeFeatureEnabledAndPassValidation>(connection, decoder, this, &TestWithValidator::enabledIfSomeFeatureEnabledAndPassValidation);
    }
    UNUSED_PARAM(connection);
    RELEASE_LOG_ERROR(IPC, "Unhandled message %s to %" PRIu64, IPC::description(decoder.messageName()).characters(), decoder.destinationID());
    decoder.markInvalid();
}

} // namespace WebKit

#if ENABLE(IPC_TESTING_API)

namespace IPC {

template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithValidator_AlwaysEnabled>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithValidator::AlwaysEnabled::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithValidator_EnabledIfPassValidation>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithValidator::EnabledIfPassValidation::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithValidator_EnabledIfSomeFeatureEnabledAndPassValidation>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithValidator::EnabledIfSomeFeatureEnabledAndPassValidation::Arguments>(globalObject, decoder);
}

}

#endif


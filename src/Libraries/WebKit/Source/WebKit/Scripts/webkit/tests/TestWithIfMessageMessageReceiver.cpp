/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 1, 2024.
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
#include "TestWithIfMessage.h"

#if PLATFORM(COCOA) || PLATFORM(GTK)
#include "ArgumentCoders.h" // NOLINT
#endif
#include "Decoder.h" // NOLINT
#include "HandleMessage.h" // NOLINT
#include "TestWithIfMessageMessages.h" // NOLINT
#if PLATFORM(COCOA) || PLATFORM(GTK)
#include <wtf/text/WTFString.h> // NOLINT
#endif

#if ENABLE(IPC_TESTING_API)
#include "JSIPCBinding.h"
#endif

namespace WebKit {

void TestWithIfMessage::didReceiveMessage(IPC::Connection& connection, IPC::Decoder& decoder)
{
    Ref protectedThis { *this };
#if PLATFORM(COCOA)
    if (decoder.messageName() == Messages::TestWithIfMessage::LoadURL::name())
        return IPC::handleMessage<Messages::TestWithIfMessage::LoadURL>(connection, decoder, this, &TestWithIfMessage::loadURL);
#endif
#if PLATFORM(GTK)
    if (decoder.messageName() == Messages::TestWithIfMessage::LoadURL::name())
        return IPC::handleMessage<Messages::TestWithIfMessage::LoadURL>(connection, decoder, this, &TestWithIfMessage::loadURL);
#endif
    UNUSED_PARAM(connection);
    RELEASE_LOG_ERROR(IPC, "Unhandled message %s to %" PRIu64, IPC::description(decoder.messageName()).characters(), decoder.destinationID());
    decoder.markInvalid();
}

} // namespace WebKit

#if ENABLE(IPC_TESTING_API)

namespace IPC {

#if PLATFORM(COCOA)
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithIfMessage_LoadURL>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithIfMessage::LoadURL::Arguments>(globalObject, decoder);
}
#endif
#if PLATFORM(GTK)
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithIfMessage_LoadURL>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithIfMessage::LoadURL::Arguments>(globalObject, decoder);
}
#endif

}

#endif


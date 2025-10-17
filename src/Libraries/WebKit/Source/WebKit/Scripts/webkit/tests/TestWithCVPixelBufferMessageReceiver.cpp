/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 15, 2022.
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
#include "TestWithCVPixelBuffer.h"

#include "Decoder.h" // NOLINT
#include "HandleMessage.h" // NOLINT
#include "TestWithCVPixelBufferMessages.h" // NOLINT
#if USE(AVFOUNDATION)
#include <WebCore/CVUtilities.h> // NOLINT
#endif
#if USE(AVFOUNDATION)
#include <wtf/RetainPtr.h> // NOLINT
#endif

#if ENABLE(IPC_TESTING_API)
#include "JSIPCBinding.h"
#endif

namespace WebKit {

void TestWithCVPixelBuffer::didReceiveMessage(IPC::Connection& connection, IPC::Decoder& decoder)
{
    Ref protectedThis { *this };
#if USE(AVFOUNDATION)
    if (decoder.messageName() == Messages::TestWithCVPixelBuffer::SendCVPixelBuffer::name())
        return IPC::handleMessage<Messages::TestWithCVPixelBuffer::SendCVPixelBuffer>(connection, decoder, this, &TestWithCVPixelBuffer::sendCVPixelBuffer);
    if (decoder.messageName() == Messages::TestWithCVPixelBuffer::ReceiveCVPixelBuffer::name())
        return IPC::handleMessageAsync<Messages::TestWithCVPixelBuffer::ReceiveCVPixelBuffer>(connection, decoder, this, &TestWithCVPixelBuffer::receiveCVPixelBuffer);
#endif
    UNUSED_PARAM(connection);
    RELEASE_LOG_ERROR(IPC, "Unhandled message %s to %" PRIu64, IPC::description(decoder.messageName()).characters(), decoder.destinationID());
    decoder.markInvalid();
}

} // namespace WebKit

#if ENABLE(IPC_TESTING_API)

namespace IPC {

#if USE(AVFOUNDATION)
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithCVPixelBuffer_SendCVPixelBuffer>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithCVPixelBuffer::SendCVPixelBuffer::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessage<MessageName::TestWithCVPixelBuffer_ReceiveCVPixelBuffer>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithCVPixelBuffer::ReceiveCVPixelBuffer::Arguments>(globalObject, decoder);
}
template<> std::optional<JSC::JSValue> jsValueForDecodedMessageReply<MessageName::TestWithCVPixelBuffer_ReceiveCVPixelBuffer>(JSC::JSGlobalObject* globalObject, Decoder& decoder)
{
    return jsValueForDecodedArguments<Messages::TestWithCVPixelBuffer::ReceiveCVPixelBuffer::ReplyArguments>(globalObject, decoder);
}
#endif

}

#endif


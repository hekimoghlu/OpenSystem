/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 9, 2022.
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
#include "XPCUtilities.h"

#if USE(EXIT_XPC_MESSAGE_WORKAROUND)
#include "Logging.h"
#include <wtf/OSObjectPtr.h>
#include <wtf/WTFProcess.h>
#include <wtf/text/ASCIILiteral.h>
#endif

namespace WebKit {

#if USE(EXIT_XPC_MESSAGE_WORKAROUND)
static constexpr auto messageNameKey = "message-name"_s;
static constexpr auto exitProcessMessage = "exit"_s;
#endif

#if !USE(EXTENSIONKIT_PROCESS_TERMINATION)
void terminateWithReason(xpc_connection_t connection, ReasonCode, const char*)
{
    // This could use ReasonSPI.h, but currently does not as the SPI is blocked by the sandbox.
    // See https://bugs.webkit.org/show_bug.cgi?id=224499 rdar://76396241
    if (!connection)
        return;

#if USE(EXIT_XPC_MESSAGE_WORKAROUND)
    // Give the process a chance to exit cleanly by sending a XPC message to request termination, then try xpc_connection_kill.
    auto exitMessage = adoptOSObject(xpc_dictionary_create(nullptr, nullptr, 0));
    xpc_dictionary_set_string(exitMessage.get(), messageNameKey, exitProcessMessage.characters());
    xpc_connection_send_message(connection, exitMessage.get());
#endif

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    xpc_connection_kill(connection, SIGKILL);
ALLOW_DEPRECATED_DECLARATIONS_END
}
#endif // !USE(EXTENSIONKIT_PROCESS_TERMINATION)

#if USE(EXIT_XPC_MESSAGE_WORKAROUND)
void handleXPCExitMessage(xpc_object_t event)
{
    if (xpc_get_type(event) == XPC_TYPE_DICTIONARY) {
        String messageName = xpc_dictionary_get_wtfstring(event, messageNameKey);
        if (messageName == exitProcessMessage) {
            RELEASE_LOG_ERROR(IPC, "Received exit message, exiting now.");
            terminateProcess(EXIT_FAILURE);
        }
    }
}
#endif // USE(EXIT_XPC_MESSAGE_WORKAROUND)

}

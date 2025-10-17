/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 14, 2024.
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
#import "config.h"
#import "XPCEndpoint.h"
#import "XPCUtilities.h"

#import <wtf/cocoa/Entitlements.h>
#import <wtf/text/ASCIILiteral.h>

#if PLATFORM(MAC)
#import "CodeSigning.h"
#import <wtf/RetainPtr.h>
#import <wtf/text/WTFString.h>
#endif

namespace WebKit {

XPCEndpoint::XPCEndpoint()
{
    m_connection = adoptOSObject(xpc_connection_create(nullptr, nullptr));
    m_endpoint = adoptOSObject(xpc_endpoint_create(m_connection.get()));

    xpc_connection_set_target_queue(m_connection.get(), dispatch_get_main_queue());
    xpc_connection_set_event_handler(m_connection.get(), ^(xpc_object_t message) {
        xpc_type_t type = xpc_get_type(message);
#if USE(EXIT_XPC_MESSAGE_WORKAROUND)
        handleXPCExitMessage(message);
#endif
        if (type == XPC_TYPE_CONNECTION) {
            OSObjectPtr<xpc_connection_t> connection = message;
#if USE(APPLE_INTERNAL_SDK)
            auto pid = xpc_connection_get_pid(connection.get());

            if (pid != getpid() && !WTF::hasEntitlement(connection.get(), "com.apple.private.webkit.use-xpc-endpoint"_s)) {
                WTFLogAlways("Audit token does not have required entitlement com.apple.private.webkit.use-xpc-endpoint");
#if PLATFORM(MAC)
                auto [signingIdentifier, isPlatformBinary] = codeSigningIdentifierAndPlatformBinaryStatus(connection.get());

                if (!isPlatformBinary || !signingIdentifier.startsWith("com.apple.WebKit.WebContent"_s)) {
                    WTFLogAlways("XPC endpoint denied to connect with unknown client");
                    return;
                }
#else
                return;
#endif
            }
#endif // USE(APPLE_INTERNAL_SDK)
            xpc_connection_set_target_queue(connection.get(), dispatch_get_main_queue());
            xpc_connection_set_event_handler(connection.get(), ^(xpc_object_t event) {
                handleEvent(connection.get(), event);
            });
            xpc_connection_resume(connection.get());
        }
    });
    xpc_connection_resume(m_connection.get());
}

void XPCEndpoint::sendEndpointToConnection(xpc_connection_t connection)
{
    if (!connection)
        return;

    auto message = adoptOSObject(xpc_dictionary_create(nullptr, nullptr, 0));
    xpc_dictionary_set_string(message.get(), xpcEndpointMessageNameKey(), xpcEndpointMessageName());
    xpc_dictionary_set_value(message.get(), xpcEndpointNameKey(), m_endpoint.get());

    xpc_connection_send_message(connection, message.get());
}

OSObjectPtr<xpc_endpoint_t> XPCEndpoint::endpoint() const
{
    return m_endpoint;
}

}

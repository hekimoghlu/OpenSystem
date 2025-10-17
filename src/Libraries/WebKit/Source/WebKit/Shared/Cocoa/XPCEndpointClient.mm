/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 14, 2023.
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
#import "XPCEndpointClient.h"

#import "Logging.h"
#import <wtf/cocoa/Entitlements.h>
#import <wtf/spi/darwin/XPCSPI.h>
#import <wtf/text/ASCIILiteral.h>

namespace WebKit {

void XPCEndpointClient::setEndpoint(xpc_endpoint_t endpoint)
{
    {
        Locker locker { m_connectionLock };

        if (m_connection)
            return;

        m_connection = adoptOSObject(xpc_connection_create_from_endpoint(endpoint));

        xpc_connection_set_target_queue(m_connection.get(), dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0));
        xpc_connection_set_event_handler(m_connection.get(), ^(xpc_object_t message) {
            xpc_type_t type = xpc_get_type(message);
            if (type == XPC_TYPE_ERROR) {
                if (message == XPC_ERROR_CONNECTION_INVALID || message == XPC_ERROR_TERMINATION_IMMINENT || XPC_ERROR_CONNECTION_INTERRUPTED) {
                    Locker locker { m_connectionLock };
                    m_connection = nullptr;
                }
                return;
            }
            if (type != XPC_TYPE_DICTIONARY)
                return;

            auto connection = xpc_dictionary_get_remote_connection(message);
            if (!connection)
                return;
#if USE(APPLE_INTERNAL_SDK)
            auto pid = xpc_connection_get_pid(connection);
            if (pid != getpid() && !WTF::hasEntitlement(connection, "com.apple.private.webkit.use-xpc-endpoint"_s)) {
                WTFLogAlways("Audit token does not have required entitlement com.apple.private.webkit.use-xpc-endpoint");
                return;
            }
#endif
            handleEvent(message);
        });

        xpc_connection_resume(m_connection.get());
    }

    didConnect();
}

OSObjectPtr<xpc_connection_t> XPCEndpointClient::connection()
{
    Locker locker { m_connectionLock };
    return m_connection;
}

}

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 3, 2024.
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
#import "DaemonUtilities.h"

#import "Encoder.h"
#import <wtf/RetainPtr.h>
#import <wtf/UniqueRef.h>
#import <wtf/cocoa/Entitlements.h>
#import <wtf/cocoa/VectorCocoa.h>
#import <wtf/text/ASCIILiteral.h>

namespace WebKit {

void startListeningForMachServiceConnections(const char* serviceName, ASCIILiteral entitlement, void(*connectionAdded)(xpc_connection_t), void(*connectionRemoved)(xpc_connection_t), void(*eventHandler)(xpc_object_t))
{
    static NeverDestroyed<RetainPtr<xpc_connection_t>> listener = xpc_connection_create_mach_service(serviceName, dispatch_get_main_queue(), XPC_CONNECTION_MACH_SERVICE_LISTENER);
    xpc_connection_set_event_handler(listener.get().get(), ^(xpc_object_t peer) {
        if (xpc_get_type(peer) != XPC_TYPE_CONNECTION)
            return;

#if USE(APPLE_INTERNAL_SDK)
        if (!entitlement.isNull() && !WTF::hasEntitlement(peer, entitlement)) {
            NSLog(@"Connection attempted without required entitlement");
            xpc_connection_cancel(peer);
            return;
        }
#endif

        xpc_connection_set_event_handler(peer, ^(xpc_object_t event) {
            if (event == XPC_ERROR_CONNECTION_INVALID) {
#if HAVE(XPC_CONNECTION_COPY_INVALIDATION_REASON)
                auto reason = std::unique_ptr<char[]>(xpc_connection_copy_invalidation_reason(peer));
                NSLog(@"Failed to start listening for connections to mach service %s, reason: %s", serviceName, reason.get());
#else
                NSLog(@"Failed to start listening for connections to mach service %s, likely because it is not registered with launchd", serviceName);
#endif
                NSLog(@"Removing peer connection %p", peer);
                connectionRemoved(peer);
                return;
            }
            if (event == XPC_ERROR_CONNECTION_INTERRUPTED) {
                NSLog(@"Removing peer connection %p", peer);
                connectionRemoved(peer);
                return;
            }
            eventHandler(event);
        });
        xpc_connection_set_target_queue(peer, dispatch_get_main_queue());
        xpc_connection_activate(peer);

        NSLog(@"Adding peer connection %p", peer);
        connectionAdded(peer);
    });
    xpc_connection_activate(listener.get().get());
}

RetainPtr<xpc_object_t> vectorToXPCData(Vector<uint8_t>&& vector)
{
    return adoptNS(xpc_data_create_with_dispatch_data(makeDispatchData(WTFMove(vector)).get()));
}

OSObjectPtr<xpc_object_t> encoderToXPCData(UniqueRef<IPC::Encoder>&& encoder)
{
    __block auto blockEncoder = WTFMove(encoder);
    auto buffer = blockEncoder->span();
    auto dispatchData = adoptNS(dispatch_data_create(buffer.data(), buffer.size(), dispatch_get_main_queue(), ^{
        // Explicitly clear out the encoder, destroying it.
        blockEncoder.moveToUniquePtr();
    }));

    return adoptOSObject(xpc_data_create_with_dispatch_data(dispatchData.get()));
}

} // namespace WebKit

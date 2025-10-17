/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 4, 2022.
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
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <Foundation/NSXPCConnection_Private.h>

#import "utilities/debugging.h"
#import <Security/SecEntitlements.h>
#import "keychain/escrowrequest/EscrowRequestServer.h"
#import "keychain/escrowrequest/EscrowRequestXPCServer.h"
#import "keychain/escrowrequest/EscrowRequestXPCProtocol.h"
#import "keychain/categories/NSError+UsefulConstructors.h"

@interface EscrowRequestXPCServer : NSObject <NSXPCListenerDelegate>
@end

@implementation EscrowRequestXPCServer

- (BOOL)listener:(__unused NSXPCListener *)listener shouldAcceptNewConnection:(NSXPCConnection *)newConnection
{
    NSNumber *num = [newConnection valueForEntitlement:kSecEntitlementPrivateEscrowRequest];
    if (![num isKindOfClass:[NSNumber class]] || ![num boolValue]) {
        secerror("escrow-update: Client pid: %d doesn't have entitlement: %@",
                 [newConnection processIdentifier], kSecEntitlementPrivateEscrowRequest);
        return NO;
    }

    secnotice("escrowrequest", "received connection from client pid %d (euid %u)", newConnection.processIdentifier, newConnection.effectiveUserIdentifier);
    newConnection.exportedInterface = SecEscrowRequestSetupControlProtocol([NSXPCInterface interfaceWithProtocol:@protocol(EscrowRequestXPCProtocol)]);

    newConnection.exportedObject = [EscrowRequestServer server];

    [newConnection resume];

    return YES;
}
@end

void
EscrowRequestXPCServerInitialize(void)
{
    static dispatch_once_t once;
    static EscrowRequestXPCServer *server;
    static NSXPCListener *listener;

    dispatch_once(&once, ^{
        @autoreleasepool {
            server = [EscrowRequestXPCServer new];

            listener = [[NSXPCListener alloc] initWithMachServiceName:@(kSecuritydEscrowRequestServiceName)];
            listener.delegate = server;
            [listener resume];
        }
    });
}

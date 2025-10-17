/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 16, 2022.
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
#import <Foundation/NSXPCConnection_Private.h>

#import "SecEntitlements.h"
#import "keychain/ckks/CKKS.h"
#import "keychain/ckks/CKKSControlProtocol.h"
#import "keychain/ckks/CKKSControlServer.h"
#import "keychain/ckks/CKKSViewManager.h"

@interface CKKSControlServer : NSObject <NSXPCListenerDelegate>
@end

@implementation CKKSControlServer

- (BOOL)listener:(__unused NSXPCListener *)listener shouldAcceptNewConnection:(NSXPCConnection *)newConnection {
#if OCTAGON
    NSNumber *num = [newConnection valueForEntitlement:(__bridge NSString *)kSecEntitlementPrivateCKKS];
    if (![num isKindOfClass:[NSNumber class]] || ![num boolValue]) {
        ckkserror_global("ckks", "Client pid: %d doesn't have entitlement: %@",
                [newConnection processIdentifier], kSecEntitlementPrivateCKKS);
        return NO;
    }

    // In the future, we should consider vending a proxy object that can return a nicer error.
    if (!SecCKKSIsEnabled()) {
        ckkserror_global("ckks", "Client pid: %d attempted to use CKKS, but CKKS is not enabled.",
                newConnection.processIdentifier);
        return NO;
    }

    newConnection.exportedInterface = CKKSSetupControlProtocol([NSXPCInterface interfaceWithProtocol:@protocol(CKKSControlProtocol)]);
    newConnection.exportedObject = [CKKSViewManager manager];

    [newConnection resume];

    return YES;
#else
    return NO;
#endif /* OCTAGON */
}

@end

void
CKKSControlServerInitialize(void)
{
    static dispatch_once_t once;
    static CKKSControlServer *server;
    static NSXPCListener *listener;

    dispatch_once(&once, ^{
        @autoreleasepool {
            server = [CKKSControlServer new];

            listener = [[NSXPCListener alloc] initWithMachServiceName:@(kSecuritydCKKSServiceName)];
            listener.delegate = server;
            [listener resume];
        }
    });
}

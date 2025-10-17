/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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

//
//  KeychainDBMoverHelpers.m
//  Security
//

#import <Foundation/Foundation.h>
#import "keychain/KeychainDBMover/KeychainDBMoverHelpers.h"
#import "keychain/KeychainDBMover/KeychainDBMover.h"
#import "debugging.h"

OSStatus SecKeychainMoveUserDb(void) {
    __block OSStatus status = errSecServiceNotAvailable;

    NSXPCConnection* moverCxn = [[NSXPCConnection alloc] initWithServiceName:@"com.apple.security.KeychainDBMover"];
    secnotice("SecKeychainMoveUserDb", "moverCxn: %@", moverCxn);
    moverCxn.remoteObjectInterface = [NSXPCInterface interfaceWithProtocol:@protocol(KeychainDBMoverProtocol)];
    [moverCxn resume];
    secdebug("SecKeychainMoveUserDb", "moverCxn resumed");

    [[moverCxn synchronousRemoteObjectProxyWithErrorHandler:^(NSError *err) {
        secerror("SecKeychainMoveUserDb: remote object failed with error: %@", err);
        status = (int)[err code];
    }] moveUserDbWithReply:^(NSError *err) {
        if (err) {
            secerror("SecKeychainMoveUserDb: replied with error: %@", err);
            status = (int)[err code];
        } else {
            status = errSecSuccess;
        }
    }];

    secdebug("SecKeychainMoveUserDb", "invalidating");
    [moverCxn invalidate];

    secnotice("SecKeychainMoveUserDb", "returning %d", (int)status);
    return status;
}

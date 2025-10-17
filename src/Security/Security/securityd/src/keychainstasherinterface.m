/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 12, 2022.
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
#import <Foundation/NSData_Private.h>

#include "utilities/debugging.h"

#import "KeychainStasherProtocol.h"
#import "keychainstasherinterface.h"

NSString* const KeychainStasherMachServiceName = @"com.apple.security.KeychainStasher";

OSStatus stashKeyWithStashAgent(uid_t client, void const* keybytes, size_t keylen) {
    if (!keybytes || keylen == 0) {
        secerror("KeychainStasherInterface: No or truncated key, won't stash");
        return errSecParam;
    }

    secnotice("KeychainStasherInterface", "Reaching out to agent to stash key");
    __block OSStatus result = errSecInternalError;
    @autoreleasepool {
        NSXPCConnection* connection = [[NSXPCConnection alloc] initWithMachServiceName:KeychainStasherMachServiceName options:0];
        [connection _setTargetUserIdentifier: client];
        connection.remoteObjectInterface = [NSXPCInterface interfaceWithProtocol:@protocol(KeychainStasherProtocol)];
        [connection resume];

        id<KeychainStasherProtocol> proxy = [connection synchronousRemoteObjectProxyWithErrorHandler:^(NSError * _Nonnull error) {
            secerror("KeychainStasherInterface: errorhandler for agent called: %@", error);
            result = errSecIO;
        }];

        NSData* key = [NSData _newZeroingDataWithBytes:keybytes length:keylen];
        [proxy stashKey:key withReply:^(NSError* error) {
            if (error) {
                secerror("KeychainStasherInterface: agent failed to stash key: %@", error);
                result = (int)error.code;
            } else {
                result = errSecSuccess;
            }
        }];

        [connection invalidate];
    }

    if (result == errSecSuccess) {
        secnotice("KeychainStasherInterface", "Successfully stashed key");
    }
    return result;
}

OSStatus loadKeyFromStashAgent(uid_t client, void** keybytes, size_t* keylen) {
    if (!keybytes || !keylen) {
        secerror("KeychainStasherInterface: No outparams for key, won't load");
        return errSecParam;
    }

    secnotice("KeychainStasherInterface", "Reaching out to agent to retrieve key");
    __block OSStatus result = errSecInternalError;
    @autoreleasepool {
        NSXPCConnection* connection = [[NSXPCConnection alloc] initWithMachServiceName:KeychainStasherMachServiceName options:0];
        [connection _setTargetUserIdentifier: client];
        connection.remoteObjectInterface = [NSXPCInterface interfaceWithProtocol:@protocol(KeychainStasherProtocol)];
        [connection resume];

        id<KeychainStasherProtocol> proxy = [connection synchronousRemoteObjectProxyWithErrorHandler:^(NSError * _Nonnull error) {
            secerror("KeychainStasherInterface: errorhandler for agent called: %@", error);
            result = errSecIO;
        }];

        [proxy loadKeyWithReply:^(NSData *key, NSError *error) {
            if (!key) {
                secerror("KeychainStasherInterface: agent failed to load key: %@", error);
                result = (int)error.code;
                return;
            }
            *keybytes = calloc(1, key.length);
            memcpy(*keybytes, key.bytes, key.length);
            *keylen = key.length;
            result = errSecSuccess;
        }];

        [connection invalidate];
    }

    if (result == errSecSuccess) {
        secnotice("KeychainStasherInterface", "Successfully loaded key");
    }
    return result;
}

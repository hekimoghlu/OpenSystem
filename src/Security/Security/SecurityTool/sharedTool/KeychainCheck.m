/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 10, 2023.
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
#import "KeychainCheck.h"
#import "SFKeychainControl.h"
#import "builtin_commands.h"
#import "SOSControlHelper.h"
#import "SOSTypes.h"
#import "CKKSControlProtocol.h"
#import <Security/SecItemPriv.h>
#import <Foundation/NSXPCConnection_Private.h>

@interface KeychainCheck ()

- (void)checkKeychain;
- (void)cleanKeychain;

@end

@implementation KeychainCheck {
    NSXPCConnection* _connection;
}

- (instancetype)initWithEndpoint:(xpc_endpoint_t)endpoint
{
    if (self = [super init]) {
        NSXPCListenerEndpoint* listenerEndpoint = [[NSXPCListenerEndpoint alloc] init];
        [listenerEndpoint _setEndpoint:endpoint];
        _connection = [[NSXPCConnection alloc] initWithListenerEndpoint:listenerEndpoint];
        if (!_connection) {
            return  nil;
        }

        NSXPCInterface* interface = [NSXPCInterface interfaceWithProtocol:@protocol(SFKeychainControl)];
        _connection.remoteObjectInterface = interface;
        [_connection resume];
    }

    return self;
}

- (void)checkKeychain
{
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    [[_connection remoteObjectProxyWithErrorHandler:^(NSError* error) {
        NSLog(@"failed to communicate with server with error: %@", error);
        dispatch_semaphore_signal(semaphore);
    }] rpcFindCorruptedItemsWithReply:^(NSArray* corruptedItems, NSError* error) {
        if (error) {
            NSLog(@"error searching keychain: %@", error.localizedDescription);
        }

        if (corruptedItems.count > 0) {
            NSLog(@"found %d corrupted items", (int)corruptedItems.count);
        }
        else {
            NSLog(@"no corrupted items found");
        }

        dispatch_semaphore_signal(semaphore);
    }];

    if (dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER)) {
        NSLog(@"timed out trying to communicate with server");
    }
}

- (void)cleanKeychain
{
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    [[_connection remoteObjectProxyWithErrorHandler:^(NSError* error) {
        NSLog(@"failed to communicate with server with error: %@", error);
        dispatch_semaphore_signal(semaphore);
    }] rpcDeleteCorruptedItemsWithReply:^(bool success, NSError* error) {
        if (success) {
            NSLog(@"successfully cleaned keychain");
        }
        else {
            NSLog(@"error attempting to clean keychain: %@", error);
        }

        dispatch_semaphore_signal(semaphore);
    }];

    if (dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER)) {
        NSLog(@"timed out trying to communicate with server");
    }
}

@end

int command_keychain_check(int argc, char* const* argv)
{
    KeychainCheck* keychainCheck = [[KeychainCheck alloc] initWithEndpoint:_SecSecuritydCopyKeychainControlEndpoint(NULL)];
    [keychainCheck checkKeychain];
    return 0;
}

int command_keychain_cleanup(int argc, char* const* argv)
{
    KeychainCheck* keychainCheck = [[KeychainCheck alloc] initWithEndpoint:_SecSecuritydCopyKeychainControlEndpoint(NULL)];
    [keychainCheck cleanKeychain];
    return 0;
}

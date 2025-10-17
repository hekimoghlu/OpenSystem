/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 20, 2023.
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
#import <Foundation/NSXPCConnection.h>
#import <Foundation/NSXPCConnection_Private.h>
#include <xpc/private.h>
#include <xpc/xpc.h>

#include "ipc/securityd_client.h"
#include "ipc/server_security_helpers.h"
#include "ipc/server_entitlement_helpers.h"
#include "ipc/server_endpoint.h"

#include "keychain/securityd/SecItemServer.h"
#include <Security/SecEntitlements.h>

#pragma mark - Securityd Server

@implementation SecuritydXPCServer
@synthesize connection = _connection;

- (instancetype)initWithConnection:(NSXPCConnection *)connection
{
    if ((self = [super init])) {
        _connection = connection;

        if (!fill_security_client(&self->_client, connection.effectiveUserIdentifier, connection.auditToken)) {
            return nil;
        }
    }
    return self;
}

- (instancetype)initWithSecurityClient:(SecurityClient*) existingClient
{
    if(!existingClient) {
        return nil;
    }
    if((self = [super init])) {
        _connection = nil;

        self->_client.task                                   = CFRetainSafe(existingClient->task);
        self->_client.accessGroups                           = CFRetainSafe(existingClient->accessGroups);
#if KEYCHAIN_SUPPORTS_SYSTEM_KEYCHAIN
        self->_client.allowSystemKeychain                    = existingClient->allowSystemKeychain;
#endif
#if KEYCHAIN_SUPPORTS_EDU_MODE_MULTIUSER
        self->_client.allowSyncBubbleKeychain                = existingClient->allowSyncBubbleKeychain;
#endif
        self->_client.isNetworkExtension                     = existingClient->isNetworkExtension;
        self->_client.canAccessNetworkExtensionAccessGroups  = existingClient->canAccessNetworkExtensionAccessGroups;
        self->_client.uid                                    = existingClient->uid;
        self->_client.musr                                   = CFRetainSafe(existingClient->musr);
#if (TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR) && TARGET_HAS_KEYSTORE
        self->_client.keybag                                 = existingClient->keybag;
#endif
#if KEYCHAIN_SUPPORTS_EDU_MODE_MULTIUSER
        self->_client.inEduMode                              = existingClient->inEduMode;
#endif

#if KEYCHAIN_SUPPORTS_EDU_MODE_MULTIUSER
        self->_client.activeUser                             = existingClient->activeUser;
#endif
        self->_client.applicationIdentifier                  = CFRetainSafe(existingClient->applicationIdentifier);
        self->_client.isAppClip                              = existingClient->isAppClip;

        self->_client.allowKeychainSharing                   = existingClient->allowKeychainSharing;
    }
    return self;
}


- (bool)clientHasBooleanEntitlement: (NSString*) entitlement {
    return SecTaskGetBooleanValueForEntitlement(self->_client.task, (__bridge CFStringRef) entitlement);
}

-(void)dealloc {
    CFReleaseNull(self->_client.task);
    CFReleaseNull(self->_client.accessGroups);
    CFReleaseNull(self->_client.musr);
    CFReleaseNull(self->_client.applicationIdentifier);
}
@end


// Class to use for local dispatching of securityd xpcs. Adds capability of fake entitlements, because you don't have a real task on the other end.
@interface LocalSecuritydXPCServer : SecuritydXPCServer
@property NSMutableDictionary<NSString*, id>* fakeEntitlements;
- (instancetype)initWithSecurityClient:(SecurityClient*) existingClient fakeEntitlements:(NSDictionary<NSString*, id>*)fakeEntitlements;
@end

@implementation LocalSecuritydXPCServer
- (instancetype)initWithSecurityClient:(SecurityClient*) existingClient fakeEntitlements:(NSDictionary<NSString*, id>*)fakeEntitlements {
    if((self = [super initWithSecurityClient: existingClient])) {
        _fakeEntitlements = [fakeEntitlements mutableCopy];
    }
    return self;
}

- (bool)clientHasBooleanEntitlement: (NSString*) entitlement {
    if(self.fakeEntitlements) {
        return [self.fakeEntitlements[entitlement] isEqual: @YES];
    } else {
        return false;
    }
}
@end


#pragma mark - SecuritydXPCServerListener

// Responsible for bringing up new SecuritydXPCServer objects, and configuring them with their remote connection
@interface SecuritydXPCServerListener : NSObject <NSXPCListenerDelegate>
@property (retain,nonnull) NSXPCListener *listener;
@end

@implementation SecuritydXPCServerListener
-(instancetype)init
{
    if((self = [super init])){
        self.listener = [[NSXPCListener alloc] initWithMachServiceName:@(kSecuritydGeneralServiceName)];
        self.listener.delegate = self;
        [self.listener resume];
    }
    return self;
}

- (BOOL)listener:(__unused NSXPCListener *)listener shouldAcceptNewConnection:(NSXPCConnection *)newConnection
{
    // Anyone is allowed to get a connection to securityd, except if you have kSecEntitlementKeychainDeny entitlement
    // The SecuritydClient class _must_ check for required entitlements in each XPC handler.

    if([newConnection valueForEntitlement: (__bridge NSString*) kSecEntitlementKeychainDeny]) {
        return NO;
    }

    newConnection.exportedInterface = [NSXPCInterface interfaceWithProtocol:@protocol(SecuritydXPCProtocol)];
    // Configure the interface on the server side, too
    [SecuritydXPCClient configureSecuritydXPCProtocol: newConnection.exportedInterface];

    newConnection.exportedObject = [[SecuritydXPCServer alloc] initWithConnection:newConnection];
    [newConnection resume];

    return YES;
}
@end

void
SecCreateSecuritydXPCServer(void)
{
    static SecuritydXPCServerListener* listener = NULL;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        @autoreleasepool {
            listener = [[SecuritydXPCServerListener alloc] init];
        }
    });
}

id<SecuritydXPCProtocol> SecCreateLocalSecuritydXPCServer(void) {
    // Create a fake securitydxpcserver using the access groups of securityd and some number of fake entitlements
    SecurityClient* client = SecSecurityClientGet();

    // We know that SecuritydXPCServerListener will comply with SecuritydXPCProtocol via category, so help the compiler out
    return (id<SecuritydXPCProtocol>) [[LocalSecuritydXPCServer alloc] initWithSecurityClient: client fakeEntitlements: @{}];
}

CFTypeRef SecCreateLocalCFSecuritydXPCServer(void) {
    return CFBridgingRetain(SecCreateLocalSecuritydXPCServer());
}

void SecResetLocalSecuritydXPCFakeEntitlements(void) {
    if([(__bridge id) gSecurityd->secd_xpc_server isKindOfClass: [LocalSecuritydXPCServer class]]) {
        LocalSecuritydXPCServer* server = (__bridge LocalSecuritydXPCServer*)gSecurityd->secd_xpc_server;
        server.fakeEntitlements = [[NSMutableDictionary alloc] init];
    }
}

void SecAddLocalSecuritydXPCFakeEntitlement(CFStringRef entitlement, CFTypeRef value) {
    if([(__bridge id) gSecurityd->secd_xpc_server isKindOfClass: [LocalSecuritydXPCServer class]]) {
        LocalSecuritydXPCServer* server = (__bridge LocalSecuritydXPCServer*)gSecurityd->secd_xpc_server;
        server.fakeEntitlements[(__bridge NSString*)entitlement] = (__bridge id)value;
    }
}

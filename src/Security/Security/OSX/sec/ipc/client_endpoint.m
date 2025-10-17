/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 16, 2025.
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
#import <objc/runtime.h>
#import <utilities/debugging.h>
#import <Security/SecXPCHelper.h>
#import <Security/SecItemFetchOutOfBandPriv.h>

#include <ipc/securityd_client.h>
#include <xpc/private.h>

@implementation SecuritydXPCClient
@synthesize connection = _connection;

- (instancetype) init
{
    return [self initTargetingSession:SecuritydXPCClient_TargetSession_CURRENT];
}

- (instancetype) initTargetingSession:(SecuritydXPCClient_TargetSession)target
{
    if ((self = [super init])) {
        NSXPCInterface *interface = [NSXPCInterface interfaceWithProtocol:@protocol(SecuritydXPCProtocol)];

        self.connection = [[NSXPCConnection alloc] initWithMachServiceName:@(kSecuritydGeneralServiceName) options:0];
        if (self.connection == NULL) {
            return NULL;
        }

        self.connection.remoteObjectInterface = interface;
        [SecuritydXPCClient configureSecuritydXPCProtocol: self.connection.remoteObjectInterface];

        if (target == SecuritydXPCClient_TargetSession_FOREGROUND) {
#if TARGET_OS_OSX
            secinfo("SecuritydXPCClient", "Targeting foreground session not supported on this platform");
#else
            secinfo("SecuritydXPCClient", "Possibly targeting foreground session");
            if (xpc_user_sessions_enabled()) {
                errno_t error = 0;
                uid_t foreground_uid = xpc_user_sessions_get_foreground_uid(&error);
                if (error != 0) {
                    secerror("SecuritydXPCClient: could not get foreground uid %d", error);
                } else {
                    secinfo("SecuritydXPCClient", "Targeting foreground session for uid %d", foreground_uid);
                    xpc_connection_set_target_user_session_uid(self.connection._xpcConnection, foreground_uid);
                }
            }
#endif
        }

        [self.connection resume];
    }

    return self;
}

- (void)dealloc
{
    [self.connection invalidate];
}

- (id<SecuritydXPCProtocol>)protocolWithSync:(bool)synchronous errorHandler:(void(^)(NSError *))errorHandler
{
    if (synchronous) {
        return [self.connection synchronousRemoteObjectProxyWithErrorHandler:errorHandler];
    } else {
        return [self.connection remoteObjectProxyWithErrorHandler:errorHandler];
    }
}

+(void)configureSecuritydXPCProtocol: (NSXPCInterface*) interface {
    NSXPCInterface *rpcCallbackInterface = [NSXPCInterface interfaceWithProtocol: @protocol(SecuritydXPCCallbackProtocol)];
    [interface setInterface:rpcCallbackInterface
                forSelector:@selector(SecItemAddAndNotifyOnSync:syncCallback:complete:)
              argumentIndex:1
                    ofReply:0];

#if OCTAGON
    NSSet<Class> *errClasses = [SecXPCHelper safeErrorClasses];

    @try {
        NSSet* arrayOfCKKSCurrentItemQueries = [NSSet setWithArray:@[[NSArray class], [CKKSCurrentItemQuery class]]];
        NSSet* arrayOfCKKSCurrentItemQueryResults = [NSSet setWithArray:@[[NSArray class], [CKKSCurrentItemQueryResult class]]];
        NSSet* arrayOfCKKSPCSIdentityQueries = [NSSet setWithArray:@[[NSArray class], [CKKSPCSIdentityQuery class]]];
        NSSet* arrayOfCKKSPCSIdentityQueryResults = [NSSet setWithArray:@[[NSArray class], [CKKSPCSIdentityQueryResult class]]];

        [rpcCallbackInterface setClasses:errClasses forSelector:@selector(callCallback:error:) argumentIndex:1 ofReply:NO];

        [interface setClasses:errClasses forSelector:@selector(SecItemAddAndNotifyOnSync:
                                                               syncCallback:
                                                               complete:) argumentIndex:2 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(secItemSetCurrentItemAcrossAllDevices:
                                                               newCurrentItemHash:
                                                               accessGroup:
                                                               identifier:
                                                               viewHint:
                                                               oldCurrentItemReference:
                                                               oldCurrentItemHash:
                                                               complete:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(secItemUnsetCurrentItemsAcrossAllDevices:
                                                               identifiers:
                                                               viewHint:
                                                               complete:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(secItemFetchCurrentItemAcrossAllDevices:
                                                               identifier:
                                                               viewHint:
                                                               fetchCloudValue:
                                                               complete:) argumentIndex:2 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(secItemDigest:
                                                               accessGroup:
                                                               complete:) argumentIndex:1 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(secKeychainDeleteMultiuser:complete:) argumentIndex:1 ofReply:YES];
        
        [interface setClasses:arrayOfCKKSCurrentItemQueries forSelector:@selector(secItemFetchCurrentItemOutOfBand:forceFetch:complete:) argumentIndex:0 ofReply:NO];
        [interface setClasses:arrayOfCKKSCurrentItemQueryResults forSelector:@selector(secItemFetchCurrentItemOutOfBand:forceFetch:complete:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(secItemFetchCurrentItemOutOfBand:forceFetch:complete:) argumentIndex:1 ofReply:YES];
        
        [interface setClasses:arrayOfCKKSPCSIdentityQueries forSelector:@selector(secItemFetchPCSIdentityByKeyOutOfBand:forceFetch:complete:) argumentIndex:0 ofReply:NO];
        [interface setClasses:arrayOfCKKSPCSIdentityQueryResults forSelector:@selector(secItemFetchPCSIdentityByKeyOutOfBand:forceFetch:complete:) argumentIndex:0 ofReply:YES];
        [interface setClasses:errClasses forSelector:@selector(secItemFetchPCSIdentityByKeyOutOfBand:forceFetch:complete:) argumentIndex:1 ofReply:YES];

    }
    @catch(NSException* e) {
        secerror("Could not configure SecuritydXPCProtocol: %@", e);
        @throw e;
    }
#endif // OCTAGON
}
@end

@implementation FakeSecuritydXPCClient
- (instancetype) init
{
    self = [super init];
    return self;
}

- (id<SecuritydXPCProtocol>)protocolWithSync:(bool)synchronous errorHandler:(void(^)(NSError *))errorHandler
{
    if (gSecurityd && gSecurityd->secd_xpc_server) {
        return (__bridge id<SecuritydXPCProtocol>)gSecurityd->secd_xpc_server;
    } else {
        return NULL;
    }
}
@end

@implementation SecuritydXPCCallback
@synthesize callback = _callback;

-(instancetype)initWithCallback: (SecBoolNSErrorCallback) callback {
    if((self = [super init])) {
        _callback = callback;
    }
    return self;
}

- (void)callCallback: (bool) result error:(NSError*) error {
    self.callback(result, error);
}
@end

id<SecuritydXPCProtocol> SecuritydXPCProxyObject(bool synchronous, void (^rpcErrorHandler)(NSError *))
{
    if (gSecurityd && gSecurityd->secd_xpc_server) {
        return (__bridge id<SecuritydXPCProtocol>)gSecurityd->secd_xpc_server;
    }

    static SecuritydXPCClient* rpc;
    static dispatch_once_t onceToken;

    dispatch_once(&onceToken, ^{
        rpc = [[SecuritydXPCClient alloc] init];
    });

    if (rpc == NULL) {
        rpcErrorHandler([NSError errorWithDomain:@"securityd" code:-1 userInfo:@{ NSLocalizedDescriptionKey : @"Could not create SecuritydXPCClient" }]);
        return NULL;
    } else {
        return [rpc protocolWithSync:synchronous errorHandler:rpcErrorHandler];
    }
}

id<SecuritydXPCClientInterface> SecuritydXPCClientObject(SecuritydXPCClient_TargetSession target, void (^errorHandler)(NSError *))
{
    id<SecuritydXPCClientInterface> rpc = (gSecurityd && gSecurityd->secd_xpc_server) ? [[FakeSecuritydXPCClient alloc] init] : [[SecuritydXPCClient alloc] initTargetingSession:target];
    if (rpc == NULL) {
        errorHandler([NSError errorWithDomain:@"securityd" code:-1 userInfo:@{ NSLocalizedDescriptionKey : @"Could not create SecuritydXPCClientObject" }]);
    }
    return rpc;
}

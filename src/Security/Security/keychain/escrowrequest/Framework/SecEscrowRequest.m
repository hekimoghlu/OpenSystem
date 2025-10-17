/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 24, 2023.
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


#import "keychain/escrowrequest/Framework/SecEscrowRequest.h"

#import "keychain/escrowrequest/EscrowRequestXPCProtocol.h"
#import "keychain/escrowrequest/EscrowRequestXPCServer.h"

#import "utilities/debugging.h"

NSString* const SecEscrowRequestHavePrecord = @"have_prerecord";
NSString* const SecEscrowRequestPendingPasscode = @"pending_passcode";
NSString* const SecEscrowRequestPendingCertificate = @"pending_certificate";

NSString* const SecEscrowRequestOptionFederationMove = @"federation_move";

@interface SecEscrowRequest ()
@property NSXPCConnection *connection;
@end

@implementation SecEscrowRequest

+ (SecEscrowRequest* _Nullable)request:(NSError* _Nullable *)error
{
    NSXPCConnection* connection = [[NSXPCConnection alloc] initWithMachServiceName:@(kSecuritydEscrowRequestServiceName) options:0];

    if (connection == nil) {
        if(error) {
            *error =  [NSError errorWithDomain:@"securityd" code:-1 userInfo:@{NSLocalizedDescriptionKey: @"Couldn't create connection (no reason given)"}];
        }
        return nil;
    }

    NSXPCInterface *interface = SecEscrowRequestSetupControlProtocol([NSXPCInterface interfaceWithProtocol:@protocol(EscrowRequestXPCProtocol)]);
    connection.remoteObjectInterface = interface;
    [connection resume];

    SecEscrowRequest* c = [[SecEscrowRequest alloc] initWithConnection:connection];
    return c;
}

- (instancetype)initWithConnection:(NSXPCConnection*)connection {
    if(self = [super init]) {
        _connection = connection;
    }
    return self;
}

- (void)dealloc
{
    [self.connection invalidate];
}

// Actual implementation

- (BOOL)triggerEscrowUpdate:(NSString*)reason error:(NSError**)error
{
    return [self triggerEscrowUpdate:reason options:nil error:error];
}

- (BOOL)triggerEscrowUpdate:(NSString*)reason options:(NSDictionary*)options error:(NSError**)error
{
    __block NSError* localError = nil;

    NSXPCConnection<EscrowRequestXPCProtocol>* c = [self.connection synchronousRemoteObjectProxyWithErrorHandler:^(NSError *xpcError) {
        localError = xpcError;
        secnotice("escrow", "triggerEscrowUpdate: Failed to get XPC connection: %@", xpcError);
    }];

    [c triggerEscrowUpdate:reason
                   options:options
                     reply:^(NSError * _Nullable xpcError) {
                         localError = xpcError;

                         secnotice("escrow", "Triggered escrow update for '%@': %@", reason, xpcError);
                     }];

    if(error && localError) {
        *error = localError;
    }

    return localError == nil;
}

- (BOOL)cachePrerecord:(NSString*)uuid
   serializedPrerecord:(NSData*)prerecord
                 error:(NSError**)error
{
    __block NSError* localError = nil;

    NSXPCConnection<EscrowRequestXPCProtocol>* c = [self.connection synchronousRemoteObjectProxyWithErrorHandler:^(NSError *xpcError) {
        localError = xpcError;
        secnotice("escrow", "cachePrerecord: Failed to get XPC connection: %@", xpcError);
    }];

    [c cachePrerecord:uuid
  serializedPrerecord:prerecord
                reply:^(NSError * _Nullable xpcError) {
                    localError = xpcError;

                    secnotice("escrow", "Cached prerecord for %@: %@", uuid, xpcError);
                }];

    if(error && localError) {
        *error = localError;
    }

    return localError == nil;
}

- (NSData* _Nullable)fetchPrerecord:(NSString*)prerecordUUID
                              error:(NSError**)error
{
    __block NSError* localError = nil;

    NSXPCConnection<EscrowRequestXPCProtocol>* c = [self.connection synchronousRemoteObjectProxyWithErrorHandler:^(NSError *xpcError) {
        localError = xpcError;
        secnotice("escrow", "fetchprerecord: Failed to get XPC connection: %@", xpcError);
    }];

    __block NSData* prerecord = nil;
    [c fetchPrerecord:prerecordUUID reply:^(NSData * _Nullable requestContents, NSError * _Nullable xpcError) {
        prerecord = requestContents;
        localError = xpcError;

        secnotice("escrow", "Received prerecord for %@: %@", prerecordUUID, xpcError);
    }];

    if(error && localError) {
        *error = localError;
    }

    return prerecord;
}

- (NSString* _Nullable)fetchRequestWaitingOnPasscode:(NSError**)error
{
    __block NSError* localError = nil;

    NSXPCConnection<EscrowRequestXPCProtocol>* c = [self.connection synchronousRemoteObjectProxyWithErrorHandler:^(NSError *xpcError) {
        localError = xpcError;
        secnotice("escrow", "fetchRequestWaitingOnPasscode: Failed to get XPC connection: %@", xpcError);
    }];

    __block NSString* uuid = nil;
    [c fetchRequestWaitingOnPasscode:^(NSString * _Nullable requestUUID, NSError * _Nullable xpcError) {
        uuid = requestUUID;
        localError = xpcError;

        secnotice("escrow", "Received request waiting on passcode: %@ %@", requestUUID, xpcError);
    }];

    if(error && localError) {
        *error = localError;
    }

    return uuid;
}

- (NSDictionary * _Nullable)fetchStatuses:(NSError**)error
{
    __block NSError* localError = nil;
    __block NSDictionary<NSString*, NSString*>* statuses = nil;

    NSXPCConnection<EscrowRequestXPCProtocol>* c = [self.connection synchronousRemoteObjectProxyWithErrorHandler:^(NSError *xpcError) {
        localError = xpcError;
        secnotice("escrow", "requestStatuses: Failed to get XPC connection: %@", xpcError);
    }];

    [c fetchRequestStatuses:^(NSDictionary<NSString*,NSString*> * _Nullable fetchedStatuses, NSError * _Nullable xpcError) {
        statuses = fetchedStatuses;
        localError = xpcError;

        secnotice("escrow", "Received statuses: %@ %@", statuses, xpcError);
    }];

    if(error && localError) {
        *error = localError;
    }

    return statuses;
}

- (BOOL)resetAllRequests:(NSError**)error
{
    __block NSError* localError = nil;

    NSXPCConnection<EscrowRequestXPCProtocol>* c = [self.connection synchronousRemoteObjectProxyWithErrorHandler:^(NSError *xpcError) {
        localError = xpcError;
        secnotice("escrow", "resetAllRequests: Failed to get XPC connection: %@", xpcError);
    }];

    [c resetAllRequests:^(NSError * _Nullable xpcError) {
                        localError = xpcError;

                        secnotice("escrow", "resetAllRequests: %@", xpcError);
                    }];

    if(error && localError) {
        *error = localError;
    }

    return localError == nil;
}

- (uint64_t)storePrerecordsInEscrow:(NSError**)error
{
    __block NSError* localError = nil;

    NSXPCConnection<EscrowRequestXPCProtocol>* c = [self.connection synchronousRemoteObjectProxyWithErrorHandler:^(NSError *xpcError) {
        localError = xpcError;
        secnotice("escrow", "fetchRequestWaitingOnPasscode: Failed to get XPC connection: %@", xpcError);
    }];

    __block uint64_t count = 0;
    [c storePrerecordsInEscrow:^(uint64_t requestCount, NSError * _Nullable xpcError) {
        count = requestCount;
        localError = xpcError;

        secnotice("escrow", "Stored %d records: %@", (int)requestCount, xpcError);
    }];

    if(error && localError) {
        *error = localError;
    }

    return count;
}

- (bool)pendingEscrowUpload:(NSError**)error
{
    NSError* localError = nil;
    
    NSDictionary<NSString*, NSString*>* result = [self fetchStatuses:&localError];
    if(localError) {
        secnotice("escrow", "failed to fetch escrow statuses: %@", localError);
        if(error) {
            *error = localError;
        }
        return NO;
    }
    if(result == nil || (result && [result count] == 0)) {
        return NO;
    }

    BOOL inProgress = NO;
    for(NSString* status in result.allValues) {
        if([status isEqualToString:SecEscrowRequestHavePrecord] ||
           [status isEqualToString:SecEscrowRequestPendingPasscode] ||
           [status isEqualToString:SecEscrowRequestPendingCertificate]) {
            inProgress = YES;
        }
    }

    return inProgress;
}

- (BOOL)escrowCompletedWithinLastSeconds:(NSTimeInterval)timeInterval
{
    __block BOOL result = true;
    __block NSError* localError = nil;

    NSXPCConnection<EscrowRequestXPCProtocol>* c = [self.connection synchronousRemoteObjectProxyWithErrorHandler:^(NSError *xpcError) {
        localError = xpcError;
        secnotice("escrow", "resetAllRequests: Failed to get XPC connection: %@", xpcError);
    }];

    [c escrowCompletedWithinLastSeconds:timeInterval reply:^(BOOL escrowCompletedWithin, NSError * _Nullable xpcError) {
        result = escrowCompletedWithin;
        localError = xpcError;

        secnotice("escrow", "resetAllRequests: %@", xpcError);
    }];

    return result;
}

@end

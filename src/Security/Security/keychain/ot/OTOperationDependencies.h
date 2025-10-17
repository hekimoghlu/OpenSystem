/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 31, 2022.
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

#import "keychain/ot/CuttlefishXPCWrapper.h"
#import "keychain/ot/OctagonStateMachine.h"
#import "keychain/ot/OTSOSAdapter.h"
#import "keychain/ot/OTAuthKitAdapter.h"
#import "keychain/ot/OTAccountsAdapter.h"
#import "keychain/ot/OTPersonaAdapter.h"
#import "keychain/ot/OTCuttlefishAccountStateHolder.h"
#import "keychain/ot/OTDeviceInformationAdapter.h"
#import "keychain/ckks/CKKSKeychainView.h"
#import "keychain/ckks/CKKSNearFutureScheduler.h"
#import "keychain/TrustedPeersHelper/TrustedPeersHelperProtocol.h"
#import <Security/SecEscrowRequest.h>

NS_ASSUME_NONNULL_BEGIN

// Used for dependency injection into most OctagonStateTransition operations
@interface OTOperationDependencies : NSObject

@property NSString* containerName;
@property NSString* contextID;

@property (nullable) TPSpecificUser* activeAccount;

@property OTCuttlefishAccountStateHolder* stateHolder;

@property id<OctagonStateFlagHandler> flagHandler;
@property id<OTSOSAdapter> sosAdapter;
@property (nullable) id<CKKSPeerProvider> octagonAdapter;
@property id<OTAccountsAdapter> accountsAdapter;
@property id<OTAuthKitAdapter> authKitAdapter;
@property id<OTPersonaAdapter> personaAdapter;
@property id<OTDeviceInformationAdapter> deviceInformationAdapter;
@property (readonly) CuttlefishXPCWrapper* cuttlefishXPCWrapper;
@property (readonly, weak) CKKSKeychainView* ckks;

@property (readonly) CKKSReachabilityTracker* reachabilityTracker;

@property CKKSLockStateTracker* lockStateTracker;
@property Class<SecEscrowRequestable> escrowRequestClass;
@property Class<CKKSNotifier> notifierClass;

@property (nullable, strong) NSString* flowID;
@property (nullable, strong) NSString* deviceSessionID;
@property (nonatomic) BOOL permittedToSendMetrics;

- (instancetype)initForContainer:(NSString*)containerName
                       contextID:(NSString*)contextID
                   activeAccount:(TPSpecificUser* _Nullable)activeAccount
                     stateHolder:(OTCuttlefishAccountStateHolder*)stateHolder
                     flagHandler:(id<OctagonStateFlagHandler>)flagHandler
                      sosAdapter:(id<OTSOSAdapter>)sosAdapter
                  octagonAdapter:(id<CKKSPeerProvider> _Nullable)octagonAdapter
                 accountsAdapter:(id<OTAccountsAdapter>)accountsAdapter
                  authKitAdapter:(id<OTAuthKitAdapter>)authKitAdapter
                  personaAdapter:(id<OTPersonaAdapter>)personaAdapter
               deviceInfoAdapter:(id<OTDeviceInformationAdapter>)deviceInfoAdapter
                 ckksAccountSync:(CKKSKeychainView* _Nullable)ckks
                lockStateTracker:(CKKSLockStateTracker *)lockStateTracker
            cuttlefishXPCWrapper:(CuttlefishXPCWrapper *)cuttlefishXPCWrapper
              escrowRequestClass:(Class<SecEscrowRequestable>)escrowRequestClass
                   notifierClass:(Class<CKKSNotifier>)notifierClass
                          flowID:(NSString* _Nullable)flowID
                 deviceSessionID:(NSString* _Nullable)deviceSessionID
          permittedToSendMetrics:(BOOL)permittedToSendMetrics
             reachabilityTracker:(CKKSReachabilityTracker*)reachabilityTracker;

@end

NS_ASSUME_NONNULL_END

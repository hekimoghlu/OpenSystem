/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 30, 2021.
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
#if OCTAGON

#import <AppleFeatures/AppleFeatures.h>

#import <TrustedPeers/TrustedPeers.h>
#import "keychain/ckks/CKKSAnalytics.h"
#import "keychain/ot/categories/OTAccountMetadataClassC+KeychainSupport.h"
#import "keychain/ot/OTModifyUserControllableViewStatusOperation.h"
#import "keychain/ot/OTStates.h"
#import "keychain/ot/ObjCImprovements.h"
#import "keychain/OctagonTrust/OTNotifications.h"
#import "keychain/TrustedPeersHelper/TrustedPeersHelperProtocol.h"

@interface OTModifyUserControllableViewStatusOperation ()
@property OTOperationDependencies* deps;

@property OctagonState* peerMissingState;

@property TPPBPeerStableInfoUserControllableViewStatus intendedViewStatus;
@end

@implementation OTModifyUserControllableViewStatusOperation
@synthesize intendedState = _intendedState;
@synthesize nextState = _nextState;

- (instancetype)initWithDependencies:(OTOperationDependencies*)dependencies
                  intendedViewStatus:(TPPBPeerStableInfoUserControllableViewStatus)intendedViewStatus
                       intendedState:(OctagonState*)intendedState
                    peerMissingState:(OctagonState*)peerMissingState
                          errorState:(OctagonState*)errorState
{
    if ((self = [super init])) {
        _deps = dependencies;

        _intendedViewStatus = intendedViewStatus;

        _intendedState = intendedState;
        _peerMissingState = peerMissingState;
        _nextState = errorState;
    }
    return self;
}

- (void)groupStart
{

    if(self.intendedViewStatus == TPPBPeerStableInfoUserControllableViewStatus_FOLLOWING) {
#if TARGET_OS_WATCH || TARGET_OS_TV
        // Watches and TVs want to be able to set the FOLLOWING state
        [self performWithStatus:self.intendedViewStatus];
#else
        // For other platforms, we want to determine the actual state by asking
        WEAKIFY(self);

        // Should we ask SOS? Or Octagon?
        if(self.deps.sosAdapter.sosEnabled) {
            NSError* error = nil;
            BOOL safariViewEnabled = [self.deps.sosAdapter safariViewSyncingEnabled:&error];

            if(error) {
                secerror("octagon-ckks: Unable to fetch SOS Safari view status: %@", error);
                self.error = error;
                return;
            }

            secnotice("octagon-ckks", "Currently SOS believes the safari view is '%@'", safariViewEnabled ? @"enabled" : @"disabled");

            TPPBPeerStableInfoUserControllableViewStatus status = safariViewEnabled ?
                TPPBPeerStableInfoUserControllableViewStatus_ENABLED :
                TPPBPeerStableInfoUserControllableViewStatus_DISABLED;

            [self performWithStatus:status];
            return;
        }

        BOOL isInheritedAccount = NO;
        NSError* localError = nil;
        OTAccountMetadataClassC* currentAccountMetadata = [self.deps.stateHolder loadOrCreateAccountMetadata:&localError];

        if (!currentAccountMetadata || localError) {
            secnotice("octagon-ckks", "Failed to load account metadata: %@", localError);
        } else {
            isInheritedAccount = currentAccountMetadata.isInheritedAccount;
        }
        
        secnotice("octagon-ckks", "Determining peers' user-controllable views policy");

        [self.deps.cuttlefishXPCWrapper fetchCurrentPolicyWithSpecificUser:self.deps.activeAccount
                                                           modelIDOverride:nil
                                                        isInheritedAccount:isInheritedAccount
                                                                     reply:^(TPSyncingPolicy* _Nullable syncingPolicy,
                                                                             TPPBPeerStableInfoUserControllableViewStatus userControllableViewStatusOfPeers,
                                                                             NSError* _Nullable error) {
            STRONGIFY(self);

            if(error) {
                secnotice("octagon-ckks", "Determining peers' user-controllable views policy failed: %@", error);
                self.error = error;
                return;
            }

            secnotice("octagon-ckks", "Retrieved peers' user-controllable views policy as: %@",
                      TPPBPeerStableInfoUserControllableViewStatusAsString(userControllableViewStatusOfPeers));

            [self performWithStatus:userControllableViewStatusOfPeers];
            return;
        }];
#endif

    } else {
        [self performWithStatus:self.intendedViewStatus];
    }
}

- (void)performWithStatus:(TPPBPeerStableInfoUserControllableViewStatus)intendedViewStatus
{
    WEAKIFY(self);

    secnotice("octagon-ckks", "Setting user-controllable views to %@", TPPBPeerStableInfoUserControllableViewStatusAsString(self.intendedViewStatus));

    [self.deps.cuttlefishXPCWrapper updateWithSpecificUser:self.deps.activeAccount
                                              forceRefetch:NO
                                                deviceName:nil
                                              serialNumber:nil
                                                 osVersion:nil
                                             policyVersion:nil
                                             policySecrets:nil
                                 syncUserControllableViews:[NSNumber numberWithInt:intendedViewStatus]
                                     secureElementIdentity:nil
                                             walrusSetting:nil
                                                 webAccess:nil
                                                     reply:^(TrustedPeersHelperPeerState* peerState, TPSyncingPolicy* syncingPolicy, NSError* error) {
        STRONGIFY(self);
        if(error || !syncingPolicy) {
            secerror("octagon-ckks: setting user-controllable views status errored: %@", error);
            self.error = error;

            if([self.deps.lockStateTracker isLockedError:self.error]) {
                secnotice("octagon-ckks", "Updating user-controllable view status failed because of lock state, will retry once unlocked: %@", self.error);
                OctagonPendingFlag* pendingFlag = [[OctagonPendingFlag alloc] initWithFlag:OctagonFlagAttemptUserControllableViewStatusUpgrade
                                                                                conditions:OctagonPendingConditionsDeviceUnlocked];

                [self.deps.flagHandler handlePendingFlag:pendingFlag];
            }
            if(peerState.peerStatus & (TPPeerStatusExcluded | TPPeerStatusUnknown)) {
                secnotice("octagon-ckks", "Updating user-controllable view status failed because our self peer is excluded or missing");
                self.nextState = self.peerMissingState;
            }
            return;
        }

        secnotice("octagon-ckks", "Received syncing policy %@ with view list: %@", syncingPolicy, syncingPolicy.viewList);

        NSError* stateError = nil;
        [self.deps.stateHolder persistAccountChanges:^OTAccountMetadataClassC * _Nullable(OTAccountMetadataClassC * _Nonnull metadata) {
            [metadata setTPSyncingPolicy:syncingPolicy];
            return metadata;
        } error:&stateError];

        if(stateError) {
            secerror("octagon: failed to save policy+views: %@", stateError);
            self.error = stateError;
            return;
        }

        [self.deps.ckks setCurrentSyncingPolicy:syncingPolicy];

        [self.deps.notifierClass post:OTUserControllableViewStatusChanged];

        self.nextState = self.intendedState;
    }];
}

@end

#endif // OCTAGON

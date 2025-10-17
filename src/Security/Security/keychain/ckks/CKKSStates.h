/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 17, 2024.
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

#import <Foundation/Foundation.h>
#import "keychain/ckks/CKKSGroupOperation.h"
#import "keychain/ot/OctagonStateMachine.h"

NS_ASSUME_NONNULL_BEGIN

extern NSString* const CKKSStateTransitionErrorDomain;

typedef OctagonState CKKSState;

NSDictionary<CKKSState*, NSNumber*>* CKKSStateMap(void);
NSSet<CKKSState*>* CKKSAllStates(void);

extern CKKSState* const CKKSStateWaitForCloudKitAccountStatus;

// CKKS is currently logged out
extern CKKSState* const CKKSStateLoggedOut;

extern CKKSState* const CKKSStateWaitForTrust;
extern CKKSState* const CKKSStateLoseTrust;

extern CKKSState* const CKKSStateInitializing;
extern CKKSState* const CKKSStateInitialized;
extern CKKSState* const CKKSStateZoneCreationFailed;
extern CKKSState* const CKKSStateZoneCreationFailedDueToNetworkError;

extern CKKSState* const CKKSStateFixupRefetchCurrentItemPointers;
extern CKKSState* const CKKSStateFixupFetchTLKShares;
extern CKKSState* const CKKSStateFixupLocalReload;
extern CKKSState* const CKKSStateFixupResaveDeviceStateEntries;
extern CKKSState* const CKKSStateFixupDeleteAllCKKSTombstones;

extern CKKSState* const CKKSStateBeginFetch;
extern CKKSState* const CKKSStateFetch;
extern CKKSState* const CKKSStateFetchComplete;
extern CKKSState* const CKKSStateNeedFullRefetch;

extern CKKSState* const CKKSStateProcessReceivedKeys;

extern CKKSState* const CKKSStateCheckZoneHierarchies;

extern CKKSState* const CKKSStateProvideKeyHierarchy;
extern CKKSState* const CKKSStateProvideKeyHierarchyUntrusted;

extern CKKSState* const CKKSStateHealTLKShares;
extern CKKSState* const CKKSStateHealTLKSharesFailed;


// States to handle individual zones misbehaving
extern CKKSState* const CKKSStateTLKMissing;

extern CKKSState* const CKKSStateUnhealthy;

extern CKKSState* const CKKSStateResettingZone;
extern CKKSState* const CKKSStateResettingLocalData;
extern CKKSState* const CKKSStateZoneDeletionFailedDueToNetworkError;

extern CKKSState* const CKKSStateBecomeReady;
extern CKKSState* const CKKSStateReady;

extern CKKSState* const CKKSStateProcessIncomingQueue;
extern CKKSState* const CKKSStateRemainingClassAIncomingItems;

extern CKKSState* const CKKSStateScanLocalItems;
extern CKKSState* const CKKSStateReencryptOutgoingItems;
extern CKKSState* const CKKSStateProcessOutgoingQueue;
extern CKKSState* const CKKSStateOutgoingQueueOperationFailed;

extern CKKSState* const CKKSStateExpandToHandleAllViews;

// Fatal error. Will not proceed unless fixed from outside class.
extern CKKSState* const CKKSStateError;

// --------------------------------
// Flag initialization
typedef OctagonFlag CKKSFlag;

// The set of trusted peers has changed
extern CKKSFlag* const CKKSFlagTrustedPeersSetChanged;

extern CKKSFlag* const CKKSFlagCloudKitLoggedIn;
extern CKKSFlag* const CKKSFlagCloudKitLoggedOut;

extern CKKSFlag* const CKKSFlagBeginTrustedOperation;
extern CKKSFlag* const CKKSFlagEndTrustedOperation;

extern CKKSFlag* const CKKSFlagChangeTokenExpired;
extern CKKSFlag* const CKKSFlagCloudKitZoneMissing;

extern CKKSFlag* const CKKSFlagDeviceUnlocked;

extern CKKSFlag* const CKKSFlagFetchRequested;
// Added when a key hierarchy fetch completes.
extern CKKSFlag* const CKKSFlagFetchComplete;

extern CKKSFlag* const CKKSFlagKeyStateProcessRequested;
extern CKKSFlag* const CKKSFlagKeySetRequested;

extern CKKSFlag* const CKKSFlagCheckQueues;
extern CKKSFlag* const CKKSFlagProcessIncomingQueueWithFreshPolicy;

extern CKKSFlag* const CKKSFlagProcessIncomingQueue;
extern CKKSFlag* const CKKSFlagProcessOutgoingQueue;
extern CKKSFlag* const CKKSFlagScanLocalItems;
extern CKKSFlag* const CKKSFlagItemReencryptionNeeded;

// Used to rate-limit CK writes
extern CKKSFlag* const CKKSFlagOutgoingQueueOperationRateToken;

extern CKKSFlag* const CKKSFlagNewPriorityViews;

extern CKKSFlag* const CKKSFlag24hrNotification;

extern CKKSFlag* const CKKSFlagZoneCreation;
extern CKKSFlag* const CKKSFlagZoneDeletion;

NSSet<CKKSFlag*>* CKKSAllStateFlags(void);

NS_ASSUME_NONNULL_END

#endif

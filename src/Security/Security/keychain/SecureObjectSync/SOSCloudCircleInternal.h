/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 28, 2021.
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
#ifndef _SECURITY_SOSCLOUDCIRCLEINTERNAL_H_
#define _SECURITY_SOSCLOUDCIRCLEINTERNAL_H_

#include <Security/SecureObjectSync/SOSCloudCircle.h>
#include <Security/SecureObjectSync/SOSPeerInfo.h>
#import <Security/OTConstants.h>

#include <xpc/xpc.h>
#include <Security/SecKey.h>

// Use this for SOS SPI framework side to check if SOS compatibility mode is enabled
#define IF_SOS_DISABLED if(!SOSCCIsSOSTrustAndSyncingEnabled())

// Use this for server side SOS compatibility mode checking
#define IF_SOS_DISABLED_SERVER if(!SOSCompatibilityModeGetCachedStatus())

// This is used for any SOSEngine creation functions.
// SOSAccount is responsible for creating the engine, the circle and transports.  To make enabling/disabling compatibility mode easier, allow creation to occur outside of whether or not compatibility mode is enabled.
#define IF_PLATFORM_DOES_NOT_SUPPORTS_SOS if(!OctagonPlatformSupportsSOS())

__BEGIN_DECLS

// Use the kSecAttrViewHint* constants in SecItemPriv.h instead

extern const CFStringRef kSOSViewHintPCSMasterKey DEPRECATED_ATTRIBUTE;
extern const CFStringRef kSOSViewHintPCSiCloudDrive DEPRECATED_ATTRIBUTE;
extern const CFStringRef kSOSViewHintPCSPhotos DEPRECATED_ATTRIBUTE;
extern const CFStringRef kSOSViewHintPCSCloudKit DEPRECATED_ATTRIBUTE;
extern const CFStringRef kSOSViewHintPCSEscrow DEPRECATED_ATTRIBUTE;
extern const CFStringRef kSOSViewHintPCSFDE DEPRECATED_ATTRIBUTE;
extern const CFStringRef kSOSViewHintPCSMailDrop DEPRECATED_ATTRIBUTE;
extern const CFStringRef kSOSViewHintPCSiCloudBackup DEPRECATED_ATTRIBUTE;
extern const CFStringRef kSOSViewHintPCSNotes DEPRECATED_ATTRIBUTE;
extern const CFStringRef kSOSViewHintPCSiMessage DEPRECATED_ATTRIBUTE;

extern const CFStringRef kSOSViewHintAppleTV DEPRECATED_ATTRIBUTE;
extern const CFStringRef kSOSViewHintHomeKit DEPRECATED_ATTRIBUTE;

CFArrayRef SOSCCCopyConcurringPeerPeerInfo(CFErrorRef* error);

bool SOSCCPurgeUserCredentials(CFErrorRef* error);

CFStringRef SOSCCGetStatusDescription(SOSCCStatus status);
CFStringRef SOSCCGetViewResultDescription(SOSViewResultCode vrc);
bool SOSCCAccountHasPublicKey(CFErrorRef *error);

/*!
 @function SOSCCProcessSyncWithPeers
 @abstract Returns the peers for whom we handled syncing from the list send to us.
 @param peers Set of peerIDs to sync with
 @param backupPeers Set of backup peerIDs to sync with
 */
CFSetRef /* CFString */ SOSCCProcessSyncWithPeers(CFSetRef peers, CFSetRef backupPeers, CFErrorRef* error);

/*!
 @function SOSCCProcessSyncWithAllPeers
 @abstract Returns the information (string, hopefully URL) that will lead to an explanation of why you have an incompatible circle.
 @param error What went wrong if we returned NULL.
 */
SyncWithAllPeersReason SOSCCProcessSyncWithAllPeers(CFErrorRef* error);

bool SOSCCProcessEnsurePeerRegistration(CFErrorRef* error);

bool SOSCCCleanupKVSKeys(CFErrorRef *error);


/*!
 @function SOSCCCopyMyPeerInfo
 @abstract Returns a copy of my peer info
 @param error What went wrong if we returned NULL
 */
SOSPeerInfoRef SOSCCCopyMyPeerInfo(CFErrorRef *error);

//
// Security Tool calls
//
CFDataRef SOSCCCopyRecoveryPublicKey(CFErrorRef *error);
CFDataRef SOSCCCopyInitialSyncData(SOSInitialSyncFlags flags, CFErrorRef *error);

void SOSCCForEachEngineStateAsStringFromArray(CFArrayRef states, void (^block)(CFStringRef oneStateString));

bool SOSCCSetSOSDisabledError(CFErrorRef *error);

bool SOSCCPushResetCircle(CFErrorRef *error);

__END_DECLS

#endif

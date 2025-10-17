/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 16, 2024.
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
/*!
 @header SOSPeer
 The functions provided in SOSPeer provide an interface to a
 secure object syncing peer in a circle
 */

#ifndef _SOSPEER_H_
#define _SOSPEER_H_

#include "keychain/SecureObjectSync/SOSCoder.h"
#include "keychain/SecureObjectSync/SOSDataSource.h" // For SOSEngineRef
#include "utilities/SecAKSWrappers.h" // TODO: Layer violation -> move to datasource or make schema based

__BEGIN_DECLS

// PeerMetaRef are used to paas info about peers between accout and engine as well as to serialize in the peerstate.
typedef CFTypeRef SOSPeerMetaRef;


// peerID is not optional everything else is.
SOSPeerMetaRef SOSPeerMetaCreateWithComponents(CFStringRef peerID, CFSetRef views, CFDataRef keybag);

// peerID and state are both required.  State is excepted to contain the set of views for this peer.
SOSPeerMetaRef SOSPeerMetaCreateWithState(CFStringRef peerID, CFDictionaryRef state);

CFTypeRef SOSPeerOrStateSetViewsKeyBagAndCreateCopy(CFTypeRef peerOrState, CFSetRef views, CFDataRef keyBag);
CFTypeRef SOSPeerOrStateSetViewsAndCopyState(CFTypeRef peerOrState, CFSetRef views);
bool SOSPeerMapEntryIsBackup(const void *mapEntry);

// peerID will always be returned, views, and publicKey might not be.
CFStringRef SOSPeerMetaGetComponents(SOSPeerMetaRef peerMeta, CFSetRef *views, CFDataRef *keybag, CFErrorRef *error);


typedef struct __OpaqueSOSPeer *SOSPeerRef;

CFTypeID SOSPeerGetTypeID(void);

void SOSPeerMarkDigestsInUse(SOSPeerRef peer, struct SOSDigestVector *mdInUse);
void SOSPeerAddManifestsInUse(SOSPeerRef peer, CFMutableDictionaryRef mfc);
bool SOSPeerDidReceiveRemovalsAndAdditions(SOSPeerRef peer, SOSManifestRef absentFromRemote, SOSManifestRef additionsFromRemote,
                                           SOSManifestRef unwantedFromRemote, SOSManifestRef local, CFErrorRef *error);
bool SOSPeerDataSourceWillCommit(SOSPeerRef peer, SOSDataSourceTransactionSource source, SOSManifestRef removals, SOSManifestRef additions, CFErrorRef *error);
bool SOSPeerDataSourceWillChange(SOSPeerRef peer, SOSDataSourceRef dataSource, SOSDataSourceTransactionSource source, CFArrayRef changes, CFErrorRef *error);
bool SOSPeerWriteAddEvent(FILE *journalFile, keybag_handle_t kbhandle, SOSDataSourceRef dataSource, SOSObjectRef object, CFErrorRef *error);

// Create a peer from an archived state.
SOSPeerRef SOSPeerCreateWithState(SOSEngineRef engine, CFStringRef peer_id, CFDictionaryRef state, CFErrorRef *error);

// Return encoded peerState dictionary
CFDictionaryRef SOSPeerCopyState(SOSPeerRef peer, CFErrorRef *error);

// (Re)initialize from a peerState dictionary
bool SOSPeerSetState(SOSPeerRef peer, SOSEngineRef engine, CFDictionaryRef state, CFErrorRef *error);
void SOSPeerSetOTRTimer(SOSPeerRef peer, dispatch_source_t timer);
dispatch_source_t SOSPeerGetOTRTimer(SOSPeerRef peer);
void SOSPeerRemoveOTRTimerEntry(SOSPeerRef peer);
bool SOSPeerTimerForPeerExist(SOSPeerRef peer);
    
//
//
//

CFIndex SOSPeerGetVersion(SOSPeerRef peer);
CFStringRef SOSPeerGetID(SOSPeerRef peer);
bool SOSPeersEqual(SOSPeerRef peerA, SOSPeerRef peerB);

uint64_t SOSPeerNextSequenceNumber(SOSPeerRef peer);
uint64_t SOSPeerGetMessageVersion(SOSPeerRef peer);

//
// MARK: State tracking helpers
//

// Return true if the peer needs saving.
bool SOSPeerDidConnect(SOSPeerRef peer);
bool SOSPeerMustSendMessage(SOSPeerRef peer);
void SOSPeerSetMustSendMessage(SOSPeerRef peer, bool must);

bool SOSPeerSendObjects(SOSPeerRef peer);
void SOSPeerSetSendObjects(SOSPeerRef peer, bool sendObjects);

bool SOSPeerHasBeenInSync(SOSPeerRef peer);
void SOSPeerSetHasBeenInSync(SOSPeerRef peer, bool hasBeenInSync);

SOSManifestRef SOSPeerGetProposedManifest(SOSPeerRef peer);
SOSManifestRef SOSPeerGetConfirmedManifest(SOSPeerRef peer);
void SOSPeerSetConfirmedManifest(SOSPeerRef peer, SOSManifestRef confirmed);
void SOSPeerAddProposedManifest(SOSPeerRef peer, SOSManifestRef pending);
void SOSPeerSetProposedManifest(SOSPeerRef peer, SOSManifestRef pending);
void SOSPeerAddLocalManifest(SOSPeerRef peer, SOSManifestRef local);
SOSManifestRef SOSPeerGetPendingObjects(SOSPeerRef peer);
void SOSPeerSetPendingObjects(SOSPeerRef peer, SOSManifestRef pendingObjects);
SOSManifestRef SOSPeerGetUnwantedManifest(SOSPeerRef peer);
void SOSPeerSetUnwantedManifest(SOSPeerRef peer, SOSManifestRef unwantedManifest);

SOSManifestRef SOSPeerCopyManifestForDigest(SOSPeerRef peer, CFDataRef digest);

CFSetRef SOSPeerGetViewNameSet(SOSPeerRef peer);
void SOSPeerSetViewNameSet(SOSPeerRef peer, CFSetRef views);

CFDataRef SOSPeerGetKeyBag(SOSPeerRef peer);
void SOSPeerKeyBagDidChange(SOSPeerRef peer);
void SOSPeerSetKeyBag(SOSPeerRef peer, CFDataRef keyBag);
// Write a reset event to the journal if mustSendMessage is true.
bool SOSPeerWritePendingReset(SOSPeerRef peer, CFErrorRef *error);

//
// MARK: Backup Peers
//

// TODO: Layer violation -> move to datasource or make schema based
bool SOSPeerAppendToJournal(SOSPeerRef peer, CFErrorRef *error, void(^with)(FILE *journalFile, keybag_handle_t kbhandle));
int SOSPeerHandoffFD(SOSPeerRef peer, CFErrorRef *error);

void SOSBackupPeerPostNotification(const char *reason);

//
// MARK: RateLimiting
//
void SOSPeerSetRateLimiter(SOSPeerRef peer, CFTypeRef limiter);
CFTypeRef SOSPeerGetRateLimiter(SOSPeerRef peer);
bool SOSPeerShouldRateLimit(CFArrayRef attributes, SOSPeerRef peer);

__END_DECLS

#endif /* !_SOSPEER_H_ */

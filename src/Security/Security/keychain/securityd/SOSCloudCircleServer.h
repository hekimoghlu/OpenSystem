/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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
#ifndef _SECURITY_SOSCLOUDCIRCLESERVER_H_
#define _SECURITY_SOSCLOUDCIRCLESERVER_H_

#import <Security/SecureObjectSync/SOSCloudCircle.h>
#include "keychain/SecureObjectSync/SOSRing.h"
#import <Security/SecKey.h>
#import <xpc/xpc.h>

__BEGIN_DECLS

//
// MARK: Server versions of our SPI
//

bool SOSCCTryUserCredentials_Server(CFStringRef user_label, CFDataRef user_password, CFStringRef dsid, CFErrorRef *error);
bool SOSCCSetUserCredentials_Server(CFStringRef user_label, CFDataRef user_password, CFErrorRef *error);
bool SOSCCSetUserCredentialsAndDSID_Server(CFStringRef user_label, CFDataRef user_password, CFStringRef dsid, CFErrorRef *error);

bool SOSCCCanAuthenticate_Server(CFErrorRef *error);
bool SOSCCPurgeUserCredentials_Server(CFErrorRef *error);

SOSCCStatus SOSCCThisDeviceIsInCircle_Server(CFErrorRef *error);
bool SOSCCRequestToJoinCircle_Server(CFErrorRef* error);
bool SOSCCRequestToJoinCircleAfterRestore_Server(CFErrorRef* error);

bool SOSCCRemoveThisDeviceFromCircle_Server(CFErrorRef* error);
bool SOSCCRemovePeersFromCircle_Server(CFArrayRef peers, CFErrorRef* error);

// This will async off the notification, so it does not return a value.
void SOSCCNotifyLoggedIntoAccount_Server(void);

bool SOSCCLoggedOutOfAccount_Server(CFErrorRef *error);
bool SOSCCBailFromCircle_Server(uint64_t limit_in_seconds, CFErrorRef* error);

CFArrayRef SOSCCCopyGenerationPeerInfo_Server(CFErrorRef* error);
CFArrayRef SOSCCCopyApplicantPeerInfo_Server(CFErrorRef* error);
CFArrayRef SOSCCCopyValidPeerPeerInfo_Server(CFErrorRef* error);
bool SOSCCValidateUserPublic_Server(CFErrorRef* error);

CFArrayRef SOSCCCopyNotValidPeerPeerInfo_Server(CFErrorRef* error);
CFArrayRef SOSCCCopyRetirementPeerInfo_Server(CFErrorRef* error);
CFArrayRef SOSCCCopyViewUnawarePeerInfo_Server(CFErrorRef* error);
bool SOSCCRejectApplicants_Server(CFArrayRef applicants, CFErrorRef* error);
bool SOSCCAcceptApplicants_Server(CFArrayRef applicants, CFErrorRef* error);

CFStringRef SOSCCCopyMyPID_Server(CFErrorRef* error);
SOSPeerInfoRef SOSCCCopyMyPeerInfo_Server(CFErrorRef* error);
CFArrayRef SOSCCCopyEngineState_Server(CFErrorRef* error);

CFArrayRef SOSCCCopyPeerPeerInfo_Server(CFErrorRef* error);
CFArrayRef SOSCCCopyConcurringPeerPeerInfo_Server(CFErrorRef* error);
bool SOSCCAccountSetToNew_Server(CFErrorRef *error);
bool SOSCCResetToOffering_Server(CFErrorRef* error);
bool SOSCCResetToEmpty_Server(CFErrorRef* error);

CFBooleanRef SOSCCPeersHaveViewsEnabled_Server(CFArrayRef viewNames, CFErrorRef *error);

#if (TARGET_OS_IOS || TARGET_OS_OSX)
int SOSCCMatchOTViews_Server(void);
#endif
SOSViewResultCode SOSCCView_Server(CFStringRef view, SOSViewActionCode action, CFErrorRef *error);
bool SOSCCViewSet_Server(CFSetRef enabledViews, CFSetRef disabledViews);

enum DepartureReason SOSCCGetLastDepartureReason_Server(CFErrorRef* error);
bool SOSCCSetLastDepartureReason_Server(enum DepartureReason reason, CFErrorRef *error);

bool SOSCCProcessEnsurePeerRegistration_Server(CFErrorRef* error);

CF_RETURNS_RETAINED CFSetRef SOSCCProcessSyncWithPeers_Server(CFSetRef peers, CFSetRef backupPeers, CFErrorRef *error);
SyncWithAllPeersReason SOSCCProcessSyncWithAllPeers_Server(CFErrorRef* error);

SOSPeerInfoRef SOSCCSetNewPublicBackupKey_Server(CFDataRef newPublicBackup, CFErrorRef *error);
bool SOSCCRegisterSingleRecoverySecret_Server(CFDataRef backupSlice, bool setupV0Only, CFErrorRef *error);

bool SOSCCWaitForInitialSync_Server(CFErrorRef*);

//
// MARK: Internal kicks.
//
CF_RETURNS_RETAINED CFArrayRef SOSCCHandleUpdateMessage(CFDictionaryRef updates);


// Expected to be called when the data source changes.
void SOSCCRequestSyncWithPeer(CFStringRef peerID);
void SOSCCRequestSyncWithPeers(CFSetRef /*SOSPeerInfoRef/CFStringRef*/ peerIDs);
void SOSCCRequestSyncWithPeersList(CFArrayRef /*CFStringRef*/ peerIDs);
void SOSCCRequestSyncWithBackupPeerList(CFArrayRef /* CFStringRef */ backupPeerIDs);
bool SOSCCIsSyncPendingFor(CFStringRef peerID, CFErrorRef *error);

void SOSCCAccountTriggerSyncWithBackupPeer_server(CFStringRef peer);

void SOSCCEnsurePeerRegistration(void);
typedef void (^SOSAccountSyncablePeersBlock)(CFArrayRef trustedPeers, CFArrayRef addedPeers, CFArrayRef removedPeers);

dispatch_queue_t SOSCCGetAccountQueue(void);

CFTypeRef GetSharedAccountRef(void); // returns SOSAccount* but this header is imported by C files, so we cast through CFTypeRef

//
// MARK: Internal access to local account for tests.
//
CFTypeRef SOSKeychainAccountGetSharedAccount(void);
//
// MARK: Internal SPIs for testing
//

void SOSCCSetGestalt_Server(CFStringRef name, CFStringRef version, CFStringRef model, CFStringRef serial);
CFStringRef SOSCCCopyOSVersion(void);

//
// MARK: Testing operations, dangerous to call in normal operation.
//
bool SOSKeychainSaveAccountDataAndPurge(CFErrorRef *error);

//
// MARK: Constants for where we store persistent information in the keychain
//

extern CFStringRef kSOSAccountLabel;
extern CFStringRef kSOSPeerDataLabel;

CFDataRef SOSItemCopy(CFStringRef label, CFErrorRef* error);
bool SOSItemUpdateOrAdd(CFStringRef label, CFStringRef accessibility, CFDataRef data, CFErrorRef *error);

bool SOSCCRegisterRecoveryPublicKey_Server(CFDataRef recovery_key, CFErrorRef *error);
CFDataRef SOSCCCopyRecoveryPublicKey_Server(CFErrorRef *error);

SOSPeerInfoRef SOSCCCopyApplication_Server(CFErrorRef *error);
CFDataRef SOSCCCopyCircleJoiningBlob_Server(SOSPeerInfoRef applicant, CFErrorRef *error);
bool SOSCCJoinWithCircleJoiningBlob_Server(CFDataRef joiningBlob, PiggyBackProtocolVersion version, CFErrorRef *error);
CFDataRef SOSCCCopyInitialSyncData_Server(uint32_t flags, CFErrorRef *error);
bool SOSCCCleanupKVSKeys_Server(CFErrorRef *error);
bool SOSCCPushResetCircle_Server(CFErrorRef* error);

bool SOSCCAccountHasPublicKey_Server(CFErrorRef *error);

void sync_the_last_data_to_kvs(CFTypeRef account, bool waitForeverForSynchronization);

bool SOSCCMessageFromPeerIsPending_Server(SOSPeerInfoRef peer, CFErrorRef *error);
bool SOSCCSendToPeerIsPending_Server(SOSPeerInfoRef peer, CFErrorRef *error);

// SOS compatibility mode
bool SOSCCSetCompatibilityMode_Server(bool compatibilityMode, CFErrorRef *error);
bool SOSCCFetchCompatibilityMode_Server(CFErrorRef *error);
bool SOSCCFetchCompatibilityModeCachedValue_Server(CFErrorRef *error);
bool SOSCCIsSOSTrustAndSyncingEnabled_Server(void);
bool SOSCompatibilityModeGetCachedStatus(void);

void SOSCCPerformWithOctagonSigningKey(void (^action)(SecKeyRef octagonPrivKey, CFErrorRef error));
void SOSCCPerformWithOctagonSigningPublicKey(void (^action)(SecKeyRef octagonPublicKey, CFErrorRef error));
void SOSCCPerformWithOctagonEncryptionKey(void (^action)(SecKeyRef octagonPrivEncryptionKey, CFErrorRef error));
void SOSCCPerformWithOctagonEncryptionPublicKey(void (^action)(SecKeyRef octagonPublicEncryptionKey, CFErrorRef error));
void SOSCCPerformWithAllOctagonKeys(void (^action)(SecKeyRef octagonEncryptionKey, SecKeyRef octagonSigningKey, CFErrorRef error));
void SOSCCPerformWithTrustedPeers(void (^action)(CFSetRef sosPeerInfoRefs, CFErrorRef error));
void SOSCCPerformWithPeerID(void (^action)(CFStringRef peerID, CFErrorRef error));
void SOSCCPerformUpdateOfAllOctagonKeys(CFDataRef octagonSigningFullKey, CFDataRef octagonEncryptionFullKey,
                                        CFDataRef signingPublicKey, CFDataRef encryptionPublicKey,
                                        SecKeyRef octagonSigningPublicKeyRef, SecKeyRef octagonEncryptionPublicKeyRef,
                                        SecKeyRef octagonSigningFullKeyRef, SecKeyRef octagonEncryptionFullKeyRef,
                                        void (^action)(CFErrorRef error));
void SOSCCPerformPreloadOfAllOctagonKeys(CFDataRef octagonSigningFullKey, CFDataRef octagonEncryptionFullKey,
                                         SecKeyRef octagonSigningFullKeyRef, SecKeyRef octagonEncryptionFullKeyRef,
                                         SecKeyRef octagonSigningPublicKeyRef, SecKeyRef octagonEncryptionPublicKeyRef,
                                         void (^action)(CFErrorRef error));

bool SOSCCSetCKKS4AllStatus(bool status, CFErrorRef* error);

void SOSCCResetOTRNegotiation_Server(CFStringRef peerid);
void SOSCCPeerRateLimiterSendNextMessage_Server(CFStringRef peerid, CFStringRef accessGroup);

#if __OBJC2__
bool SOSCCSaveOctagonKeysToKeychain(NSString* keyLabel, NSData* keyDataToSave, __unused int keySize, SecKeyRef octagonPublicKey, NSError** error);
void SOSCCEnsureAccessGroupOfKey(SecKeyRef publicKey, NSString* oldAgrp, NSString* newAgrp);
#endif

// for secdtests
void enableSOSCompatibilityForTests(void);

__END_DECLS

#endif

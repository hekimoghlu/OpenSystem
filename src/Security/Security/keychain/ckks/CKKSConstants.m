/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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
#include <sys/sysctl.h>

#import "keychain/ckks/CKKS.h"

const SecCKKSItemEncryptionVersion currentCKKSItemEncryptionVersion = CKKSItemEncryptionVersion2;

NSString* const SecCKKSActionAdd = @"add";
NSString* const SecCKKSActionDelete = @"delete";
NSString* const SecCKKSActionModify = @"modify";

CKKSItemState* const SecCKKSStateNew = (CKKSItemState*) @"new";
CKKSItemState* const SecCKKSStateUnauthenticated = (CKKSItemState*) @"unauthenticated";
CKKSItemState* const SecCKKSStateInFlight = (CKKSItemState*) @"inflight";
CKKSItemState* const SecCKKSStateReencrypt = (CKKSItemState*) @"reencrypt";
CKKSItemState* const SecCKKSStateError = (CKKSItemState*) @"error";
CKKSItemState* const SecCKKSStateDeleted = (CKKSItemState*) @"deleted";
CKKSItemState* const SecCKKSStateMismatchedView = (CKKSItemState*) @"mismatched_view";

CKKSProcessedState* const SecCKKSProcessedStateLocal = (CKKSProcessedState*) @"local";
CKKSProcessedState* const SecCKKSProcessedStateRemote = (CKKSProcessedState*) @"remote";

CKKSKeyClass* const SecCKKSKeyClassTLK = (CKKSKeyClass*) @"tlk";
CKKSKeyClass* const SecCKKSKeyClassA = (CKKSKeyClass*) @"classA";
CKKSKeyClass* const SecCKKSKeyClassC = (CKKSKeyClass*) @"classC";

NSString* SecCKKSContainerName = @"com.apple.security.keychain";

// This is the empty string because that's the default value of the column, so upgraded dbs will have this set on their existing data
NSString* CKKSDefaultContextID = @"";

NSString* const SecCKKSSubscriptionID = @"keychain-changes";
NSString* const SecCKKSAPSNamedPort = @"com.apple.securityd.aps";

NSString* const SecCKRecordItemType = @"item";
NSString* const SecCKRecordHostOSVersionKey = @"uploadver";
NSString* const SecCKRecordEncryptionVersionKey = @"encver";
NSString* const SecCKRecordDataKey = @"data";
NSString* const SecCKRecordParentKeyRefKey = @"parentkeyref";
NSString* const SecCKRecordWrappedKeyKey = @"wrappedkey";
NSString* const SecCKRecordGenerationCountKey = @"gen";

NSString* const SecCKRecordPCSServiceIdentifier = @"pcsservice";
NSString* const SecCKRecordPCSPublicKey = @"pcspublickey";
NSString* const SecCKRecordPCSPublicIdentity = @"pcspublicidentity";
NSString* const SecCKRecordServerWasCurrent = @"server_wascurrent";

NSString* const SecCKRecordIntermediateKeyType = @"synckey";
NSString* const SecCKRecordKeyClassKey = @"class";

NSString* const SecCKRecordTLKShareType = @"tlkshare";
NSString* const SecCKRecordSenderPeerID = @"sender";
NSString* const SecCKRecordReceiverPeerID = @"receiver";
NSString* const SecCKRecordReceiverPublicEncryptionKey = @"receiverPublicEncryptionKey";
NSString* const SecCKRecordCurve = @"curve";
NSString* const SecCKRecordEpoch = @"epoch";
NSString* const SecCKRecordPoisoned = @"poisoned";
NSString* const SecCKRecordSignature = @"signature";
NSString* const SecCKRecordVersion = @"version";

NSString* const SecCKRecordCurrentKeyType = @"currentkey";

NSString* const SecCKRecordCurrentItemType = @"currentitem";
NSString* const SecCKRecordItemRefKey = @"item";

NSString* const SecCKRecordDeviceStateType = @"devicestate";
NSString* const SecCKRecordOctagonPeerID = @"octagonpeerid";
NSString* const SecCKRecordOctagonStatus = @"octagonstatus";
NSString* const SecCKRecordCirclePeerID = @"peerid";
NSString* const SecCKRecordCircleStatus = @"circle";
NSString* const SecCKRecordKeyState = @"keystate";
NSString* const SecCKRecordCurrentTLK = @"currentTLK";
NSString* const SecCKRecordCurrentClassA = @"currentClassA";
NSString* const SecCKRecordCurrentClassC = @"currentClassC";
NSString* const SecCKSRecordLastUnlockTime = @"lastunlock";
NSString* const SecCKSRecordOSVersionKey = @"osver";

NSString* const SecCKRecordManifestType = @"manifest";
NSString* const SecCKRecordManifestDigestValueKey = @"digest_value";
NSString* const SecCKRecordManifestGenerationCountKey = @"generation_count";
NSString* const SecCKRecordManifestLeafRecordIDsKey = @"leaf_records";
NSString* const SecCKRecordManifestPeerManifestRecordIDsKey = @"peer_manifests";
NSString* const SecCKRecordManifestCurrentItemsKey = @"current_items";
NSString* const SecCKRecordManifestSignaturesKey = @"signatures";
NSString* const SecCKRecordManifestSignerIDKey = @"signer_id";
NSString* const SecCKRecordManifestSchemaKey = @"schema";

NSString* const SecCKRecordManifestLeafType = @"manifest_leaf";
NSString* const SecCKRecordManifestLeafDERKey = @"der";
NSString* const SecCKRecordManifestLeafDigestKey = @"digest";

CKKSZoneKeyState* const SecCKKSZoneKeyStateWaitForCloudKitAccountStatus = (CKKSZoneKeyState*)@"wait_for_ck_account_status";

CKKSZoneKeyState* const SecCKKSZoneKeyStateReady = (CKKSZoneKeyState*) @"ready";
CKKSZoneKeyState* const SecCKKSZoneKeyStateBecomeReady = (CKKSZoneKeyState*) @"become_ready";
CKKSZoneKeyState* const SecCKKSZoneKeyStateReadyPendingUnlock = (CKKSZoneKeyState*) @"readypendingunlock";
CKKSZoneKeyState* const SecCKKSZoneKeyStateError = (CKKSZoneKeyState*) @"error";

CKKSZoneKeyState* const SecCKKSZoneKeyStateInitializing = (CKKSZoneKeyState*) @"initializing";
CKKSZoneKeyState* const SecCKKSZoneKeyStateInitialized = (CKKSZoneKeyState*) @"initialized";
CKKSZoneKeyState* const SecCKKSZoneKeyStateBeginFetch = (CKKSZoneKeyState*) @"begin_fetch";
CKKSZoneKeyState* const SecCKKSZoneKeyStateFetch = (CKKSZoneKeyState*) @"fetching";
CKKSZoneKeyState* const SecCKKSZoneKeyStateFetchComplete = (CKKSZoneKeyState*) @"fetchcomplete";
CKKSZoneKeyState* const SecCKKSZoneKeyStateNeedFullRefetch = (CKKSZoneKeyState*) @"needrefetch";

CKKSZoneKeyState* const SecCKKSZoneKeyStateTLKMissing = (CKKSZoneKeyState*) @"tlkmissing";
CKKSZoneKeyState* const SecCKKSZoneKeyStateWaitForTLK = (CKKSZoneKeyState*) @"waitfortlk";

CKKSZoneKeyState* const SecCKKSZoneKeyStateWaitForTLKCreation = (CKKSZoneKeyState*) @"waitfortlkcreation";
CKKSZoneKeyState* const SecCKKSZoneKeyStateWaitForTLKUpload = (CKKSZoneKeyState*) @"waitfortlkupload";
CKKSZoneKeyState* const SecCKKSZoneKeyStateWaitForUnlock = (CKKSZoneKeyState*) @"waitforunlock";
CKKSZoneKeyState* const SecCKKSZoneKeyStateLoseTrust = (CKKSZoneKeyState*) @"lose_trust";
CKKSZoneKeyState* const SecCKKSZoneKeyStateWaitForTrust = (CKKSZoneKeyState*) @"waitfortrust";
CKKSZoneKeyState* const SecCKKSZoneKeyStateUnhealthy = (CKKSZoneKeyState*) @"unhealthy";
CKKSZoneKeyState* const SecCKKSZoneKeyStateBadCurrentPointers = (CKKSZoneKeyState*) @"badcurrentpointers";
CKKSZoneKeyState* const SecCKKSZoneKeyStateNewTLKsFailed = (CKKSZoneKeyState*) @"newtlksfailed";
CKKSZoneKeyState* const SecCKKSZoneKeyStateHealTLKShares = (CKKSZoneKeyState*) @"healtlkshares";
CKKSZoneKeyState* const SecCKKSZoneKeyStateHealTLKSharesFailed = (CKKSZoneKeyState*) @"healtlksharesfailed";
CKKSZoneKeyState* const SecCKKSZoneKeyStateResettingZone = (CKKSZoneKeyState*) @"resetzone";
CKKSZoneKeyState* const SecCKKSZoneKeyStateResettingLocalData = (CKKSZoneKeyState*) @"resetlocal";
CKKSZoneKeyState* const SecCKKSZoneKeyStateLoggedOut = (CKKSZoneKeyState*) @"loggedout";
CKKSZoneKeyState* const SecCKKSZoneKeyStateZoneCreationFailed = (CKKSZoneKeyState*) @"zonecreationfailed";
CKKSZoneKeyState* const SecCKKSZoneKeyStateProcess = (CKKSZoneKeyState*) @"process_key_hierarchy";

CKKSZoneKeyState* const CKKSZoneKeyStateFixupRefetchCurrentItemPointers = (CKKSZoneKeyState*) @"fixup_fetch_cip";
CKKSZoneKeyState* const CKKSZoneKeyStateFixupFetchTLKShares = (CKKSZoneKeyState*) @"fixup_fetch_tlkshares";
CKKSZoneKeyState* const CKKSZoneKeyStateFixupLocalReload = (CKKSZoneKeyState*) @"fixup_local_reload";
CKKSZoneKeyState* const CKKSZoneKeyStateFixupResaveDeviceStateEntries = (CKKSZoneKeyState*) @"fixup_resave_cdse";
CKKSZoneKeyState* const CKKSZoneKeyStateFixupDeleteAllCKKSTombstones = (CKKSZoneKeyState*) @"fixup_delete_tombstones";

CKKSZoneKeyState* const CKKSZoneKeyStateCheckTLKShares = (CKKSZoneKeyState*) @"check_tlk_shares";

NSString* const CKKSErrorDomain = @"CKKSErrorDomain";
NSString* const CKKSServerExtensionErrorDomain = @"CKKSServerExtensionErrorDomain";

const NSUInteger SecCKKSItemPaddingBlockSize = 20;

NSString* const SecCKKSAggdPropagationDelay   = @"com.apple.security.ckks.propagationdelay";
NSString* const SecCKKSAggdPrimaryKeyConflict = @"com.apple.security.ckks.pkconflict";
NSString* const SecCKKSAggdViewKeyCount = @"com.apple.security.ckks.keycount";
NSString* const SecCKKSAggdItemReencryption = @"com.apple.security.ckks.reencrypt";

NSString* const SecCKKSUserDefaultsSuite = @"com.apple.security.ckks";

NSString* SecCKKSHostOSVersion(void)
{
#ifdef PLATFORM
    // Use complicated macro magic to get the string value passed in as preprocessor define PLATFORM.
#define PLATFORM_VALUE(f) #f
#define PLATFORM_OBJCSTR(f) @PLATFORM_VALUE(f)
    NSString* platform = (PLATFORM_OBJCSTR(PLATFORM));
#undef PLATFORM_OBJCSTR
#undef PLATFORM_VALUE
#else
    NSString* platform = "unknown";
#warning No PLATFORM defined; why?
#endif

    NSString* osversion = nil;

    // If we can get the build information from sysctl, use it.
    char release[256];
    size_t releasesize = sizeof(release);
    bool haveSysctlInfo = true;
    haveSysctlInfo &= (0 == sysctlbyname("kern.osrelease", release, &releasesize, NULL, 0));

    char version[256];
    size_t versionsize = sizeof(version);
    haveSysctlInfo &= (0 == sysctlbyname("kern.osversion", version, &versionsize, NULL, 0));

    if(haveSysctlInfo) {
        // Null-terminate for extra safety
        release[sizeof(release)-1] = '\0';
        version[sizeof(version)-1] = '\0';
        osversion = [NSString stringWithFormat:@"%s (%s)", release, version];
    }

    if(!osversion) {
        //  Otherwise, use the not-really-supported fallback.
        osversion = [[NSProcessInfo processInfo] operatingSystemVersionString];

        // subtly improve osversion (but it's okay if that does nothing)
        osversion = [osversion stringByReplacingOccurrencesOfString:@"Version" withString:@""];
    }

    return [NSString stringWithFormat:@"%@ %@", platform, osversion];
}

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 13, 2024.
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
 @header SOSBackupSliceKeyBag.h - View Bags - backup bags for views
 */

#ifndef _sec_SOSBackupSliceKeyBag_
#define _sec_SOSBackupSliceKeyBag_

#include <CoreFoundation/CoreFoundation.h>
#include <Security/SecureObjectSync/SOSPeerInfo.h>

extern CFStringRef bskbRkbgPrefix;

CFDataRef SOSRKNullKey(void);

// We don't have a portable header (particularly for the SIM) so for now we define the one type we need.
// This should be fixed when we get a portable AKS interface.
typedef int32_t bskb_keybag_handle_t;

typedef struct CF_BRIDGED_TYPE(id) __OpaqueSOSBackupSliceKeyBag *SOSBackupSliceKeyBagRef;

SOSBackupSliceKeyBagRef SOSBackupSliceKeyBagCreate(CFAllocatorRef allocator, CFSetRef peers, CFErrorRef* error);
SOSBackupSliceKeyBagRef SOSBackupSliceKeyBagCreateDirect(CFAllocatorRef allocator, CFDataRef aks_bag, CFErrorRef *error);

SOSBackupSliceKeyBagRef SOSBackupSliceKeyBagCreateWithAdditionalKeys(CFAllocatorRef allocator,
                                                                     CFSetRef /*SOSPeerInfoRef*/ peers,
                                                                     CFDictionaryRef /*CFStringRef (prefix) CFDataRef (keydata) */ additionalKeys,
                                                                     CFErrorRef* error);

SOSBackupSliceKeyBagRef SOSBackupSliceKeyBagCreateFromData(CFAllocatorRef allocator, CFDataRef data, CFErrorRef *error);

CFDataRef SOSBSKBCopyEncoded(SOSBackupSliceKeyBagRef BackupSliceKeyBag, CFErrorRef* error);

//
bool SOSBSKBIsDirect(SOSBackupSliceKeyBagRef backupSliceKeyBag);

CFSetRef SOSBSKBGetPeers(SOSBackupSliceKeyBagRef backupSliceKeyBag);

int SOSBSKBCountPeers(SOSBackupSliceKeyBagRef backupSliceKeyBag);

bool SOSBSKBPeerIsInKeyBag(SOSBackupSliceKeyBagRef backupSliceKeyBag, SOSPeerInfoRef pi);
bool SOSBKSBKeyIsInKeyBag(SOSBackupSliceKeyBagRef backupSliceKeyBag, CFDataRef publicKey);
bool SOSBKSBPeerBackupKeyIsInKeyBag(SOSBackupSliceKeyBagRef backupSliceKeyBag, SOSPeerInfoRef pi);
bool SOSBSKBAllPeersBackupKeysAreInKeyBag(SOSBackupSliceKeyBagRef backupSliceKeyBag, CFSetRef peers);
bool SOSBKSBPrefixedKeyIsInKeyBag(SOSBackupSliceKeyBagRef backupSliceKeyBag, CFStringRef prefix, CFDataRef publicKey);

// Keybag fetching
CFDataRef SOSBSKBCopyAKSBag(SOSBackupSliceKeyBagRef backupSliceKeyBag, CFErrorRef* error);


// Der encoding
const uint8_t* der_decode_BackupSliceKeyBag(CFAllocatorRef allocator,
                                  SOSBackupSliceKeyBagRef* BackupSliceKeyBag, CFErrorRef *error,
                                  const uint8_t* der, const uint8_t *der_end);

size_t der_sizeof_BackupSliceKeyBag(SOSBackupSliceKeyBagRef BackupSliceKeyBag, CFErrorRef *error);
uint8_t* der_encode_BackupSliceKeyBag(SOSBackupSliceKeyBagRef BackupSliceKeyBag, CFErrorRef *error,
                            const uint8_t *der, uint8_t *der_end);

bskb_keybag_handle_t SOSBSKBLoadLocked(SOSBackupSliceKeyBagRef backupSliceKeyBag,
                                       CFErrorRef *error);

bskb_keybag_handle_t SOSBSKBLoadAndUnlockWithPeerIDAndSecret(SOSBackupSliceKeyBagRef backupSliceKeyBag,
                                                             CFStringRef peerID, CFDataRef peerSecret,
                                                             CFErrorRef *error);

bskb_keybag_handle_t SOSBSKBLoadAndUnlockWithPeerSecret(SOSBackupSliceKeyBagRef backupSliceKeyBag,
                                                        SOSPeerInfoRef peer, CFDataRef peerSecret,
                                                        CFErrorRef *error);

bskb_keybag_handle_t SOSBSKBLoadAndUnlockWithDirectSecret(SOSBackupSliceKeyBagRef backupSliceKeyBag,
                                                          CFDataRef directSecret,
                                                          CFErrorRef *error);

bskb_keybag_handle_t SOSBSKBLoadAndUnlockWithWrappingSecret(SOSBackupSliceKeyBagRef backupSliceKeyBag,
                                                            CFDataRef wrappingSecret,
                                                            CFErrorRef *error);

// Utilities for backup keys
bool SOSBSKBIsGoodBackupPublic(CFDataRef publicKey, CFErrorRef *error);

CFDataRef SOSBSKBCopyRecoveryKey(SOSBackupSliceKeyBagRef bskb);
bool SOSBSKBHasRecoveryKey(SOSBackupSliceKeyBagRef bskb);
bool SOSBSKBHasThisRecoveryKey(SOSBackupSliceKeyBagRef bskb, CFDataRef backupKey);
void SOSBSKBRemoveRecoveryKey(SOSBackupSliceKeyBagRef bskb);

#endif /* defined(_sec_SOSBackupSliceKeyBag_) */

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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
#ifndef _SOSFULLPEERINFO_H_
#define _SOSFULLPEERINFO_H_

#include <CoreFoundation/CoreFoundation.h>
#include <Security/SecKey.h>
#include <CommonCrypto/CommonDigestSPI.h>
#include <Security/SecureObjectSync/SOSPeerInfo.h>
#include <Security/SecureObjectSync/SOSCloudCircle.h>

__BEGIN_DECLS

typedef struct __OpaqueSOSFullPeerInfo   *SOSFullPeerInfoRef;

enum {
    kSOSFullPeerVersion = 1,
};

SOSFullPeerInfoRef SOSFullPeerInfoCreate(CFAllocatorRef allocator, CFDictionaryRef gestalt, CFStringRef circleName, CFDataRef backupKey, SecKeyRef signingKey, SecKeyRef octagonSigningKey, SecKeyRef octagonEncryptionKey, CFErrorRef *error);

bool SOSFullPeerInfoUpdateToThisPeer(SOSFullPeerInfoRef peer, SOSPeerInfoRef pi, CFErrorRef *error);

SOSFullPeerInfoRef SOSFullPeerInfoCreateWithViews(CFAllocatorRef allocator, CFStringRef circleName,
                                                  CFDictionaryRef gestalt, CFDataRef backupKey, CFSetRef enabledViews,
                                                  SecKeyRef signingKey, SecKeyRef octagonSigningKey, SecKeyRef octagonEncryptionKey, CFErrorRef *error);

SOSFullPeerInfoRef SOSFullPeerInfoCopyFullPeerInfo(SOSFullPeerInfoRef toCopy);

SOSFullPeerInfoRef SOSFullPeerInfoCreateCloudIdentity(CFAllocatorRef allocator, SOSPeerInfoRef peer, CFErrorRef* error);

SOSPeerInfoRef SOSFullPeerInfoGetPeerInfo(SOSFullPeerInfoRef fullPeer);
SecKeyRef      SOSFullPeerInfoCopyDeviceKey(SOSFullPeerInfoRef fullPeer, CFErrorRef* error);

CF_RETURNS_RETAINED
SecKeyRef
SOSFullPeerInfoCopyPubKey(SOSFullPeerInfoRef fpi, CFErrorRef *error);

/* octagon keys */
SecKeyRef SOSFullPeerInfoCopyOctagonPublicSigningKey(SOSFullPeerInfoRef fullPeer, CFErrorRef* error);
SecKeyRef SOSFullPeerInfoCopyOctagonPublicEncryptionKey(SOSFullPeerInfoRef fullPeer, CFErrorRef* error);
SecKeyRef SOSFullPeerInfoCopyOctagonSigningKey(SOSFullPeerInfoRef fullPeer, CFErrorRef* error);
SecKeyRef SOSFullPeerInfoCopyOctagonEncryptionKey(SOSFullPeerInfoRef fullPeer, CFErrorRef* error);
bool SOSFullPeerInfoSetCKKS4AllSupport(SOSFullPeerInfoRef fullPeerInfo, bool support, CFErrorRef* error);

bool SOSFullPeerInfoPurgePersistentKey(SOSFullPeerInfoRef peer, CFErrorRef* error);

SOSPeerInfoRef SOSFullPeerInfoPromoteToRetiredAndCopy(SOSFullPeerInfoRef peer, CFErrorRef *error);

bool SOSFullPeerInfoPing(SOSFullPeerInfoRef peer, CFErrorRef* error);

bool SOSFullPeerInfoValidate(SOSFullPeerInfoRef peer, CFErrorRef* error);

bool SOSFullPeerInfoPrivKeyExists(SOSFullPeerInfoRef peer);

bool SOSFullPeerInfoUpdateGestalt(SOSFullPeerInfoRef peer, CFDictionaryRef gestalt, CFErrorRef* error);

bool SOSFullPeerInfoUpdateV2Dictionary(SOSFullPeerInfoRef peer, CFDictionaryRef newv2dict, CFErrorRef* error);

bool SOSFullPeerInfoUpdateBackupKey(SOSFullPeerInfoRef peer, CFDataRef backupKey, CFErrorRef* error);

bool SOSFullPeerInfoReplaceEscrowRecords(SOSFullPeerInfoRef peer, CFDictionaryRef escrowRecords, CFErrorRef* error);

bool SOSFullPeerInfoUpdateToCurrent(SOSFullPeerInfoRef peer, CFSetRef minimumViews, CFSetRef excludedViews);

SOSViewResultCode SOSFullPeerInfoUpdateViews(SOSFullPeerInfoRef peer, SOSViewActionCode action, CFStringRef viewname, CFErrorRef* error);

SOSViewResultCode SOSFullPeerInfoViewStatus(SOSFullPeerInfoRef peer, CFStringRef viewname, CFErrorRef *error);

bool SOSFullPeerInfoPromoteToApplication(SOSFullPeerInfoRef fpi, SecKeyRef user_key, CFErrorRef *error);

bool SOSFullPeerInfoUpgradeSignatures(SOSFullPeerInfoRef fpi, SecKeyRef user_key, CFErrorRef *error);

//
// DER Import Export
//
SOSFullPeerInfoRef SOSFullPeerInfoCreateFromDER(CFAllocatorRef allocator, CFErrorRef* error,
                                        const uint8_t** der_p, const uint8_t *der_end);

SOSFullPeerInfoRef SOSFullPeerInfoCreateFromData(CFAllocatorRef allocator, CFDataRef fullPeerData, CFErrorRef *error);

size_t      SOSFullPeerInfoGetDEREncodedSize(SOSFullPeerInfoRef peer, CFErrorRef *error);
uint8_t*    SOSFullPeerInfoEncodeToDER(SOSFullPeerInfoRef peer, CFErrorRef* error,
                                   const uint8_t* der, uint8_t* der_end);

CFDataRef SOSFullPeerInfoCopyEncodedData(SOSFullPeerInfoRef peer, CFAllocatorRef allocator, CFErrorRef *error);

bool SOSFullPeerInfoUpdateOctagonSigningKey(SOSFullPeerInfoRef peer, SecKeyRef octagonSigningKey, CFErrorRef* error);
bool SOSFullPeerInfoUpdateOctagonEncryptionKey(SOSFullPeerInfoRef peer, SecKeyRef octagonEncryptionKey, CFErrorRef* error);
bool SOSFullPeerInfoUpdateOctagonKeys(SOSFullPeerInfoRef peer, SecKeyRef octagonSigningKey, SecKeyRef octagonEncryptionKey, CFErrorRef* error);

CFDataRef SOSPeerInfoCopyData(SOSPeerInfoRef fpi, CFErrorRef *error);

bool SOSFullPeerInfoUpdate(SOSFullPeerInfoRef fullPeerInfo, CFErrorRef *error, SOSPeerInfoRef (^create_modification)(SOSPeerInfoRef peer, SecKeyRef key, CFErrorRef *error));

__END_DECLS

#endif

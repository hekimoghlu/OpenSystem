/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 12, 2024.
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
#include <Security/SecureObjectSync/SOSBackupSliceKeyBag.h>
#include "keychain/SecureObjectSync/SOSPeerInfoCollections.h"
#include <utilities/SecCFWrappers.h>
#include <Security/SecRandom.h>

#include "keychain/SecureObjectSync/SOSInternal.h"

#include "SOSCircle_regressions.h"
#include "SOSRegressionUtilities.h"

#if SOS_ENABLED

#define encode_decode_count 2

static CF_RETURNS_RETAINED SOSBackupSliceKeyBagRef EncodeDecode(SOSBackupSliceKeyBagRef bag)
{
    SOSBackupSliceKeyBagRef result = NULL;
    CFErrorRef localError = NULL;
    SKIP: {
        skip("No bag to use", 2, bag);
        CFDataRef encoded = SOSBSKBCopyEncoded(bag, &localError);
        ok(encoded, "encode (%@)", localError);
        CFReleaseNull(localError);

        skip("Encode failed", 1, encoded);
        result = SOSBackupSliceKeyBagCreateFromData(kCFAllocatorDefault, encoded, &localError);
        ok(result, "create (%@)", localError);
        CFReleaseNull(localError);

        CFReleaseNull(encoded);
    }

    return result;
}

static const uint8_t sEntropy1[] = {
    0xc4, 0xb9, 0xa6, 0x6e, 0xeb, 0x56, 0xa1, 0x5c, 0x1d, 0x30, 0x09, 0x40,
    0x41, 0xe9, 0x68, 0xb4, 0x12, 0xe0, 0xc6, 0x69, 0xfb, 0xdf, 0xcb, 0xe0,
    0x27, 0x4b, 0x54, 0xf0, 0xdd, 0x62, 0x10, 0x78
};

static const uint8_t sEntropy2[] = {
    0xef, 0xbd, 0x72, 0x57, 0x02, 0xe6, 0xbd, 0x0a, 0x22, 0x6e, 0x77, 0x93,
    0x17, 0xb3, 0x27, 0x12, 0x1b, 0x1f, 0xdf, 0xa0, 0x5b, 0xc6, 0x66, 0x54,
    0x3a, 0x91, 0x0d, 0xc1, 0x5f, 0x57, 0x98, 0x44
};

static void tests(void)
{
    CFErrorRef localError = NULL;
    CFMutableSetRef piSet = CFSetCreateMutableForSOSPeerInfosByID(kCFAllocatorDefault);

    CFDataRef entropy1 = CFDataCreateWithBytesNoCopy(kCFAllocatorDefault, sEntropy1, sizeof(sEntropy1), kCFAllocatorNull);
    CFDataRef entropy2 = CFDataCreateWithBytesNoCopy(kCFAllocatorDefault, sEntropy2, sizeof(sEntropy2), kCFAllocatorNull);

    SecKeyRef peer1SigningKey = NULL;
    SecKeyRef peer1OctagonSigningKey = NULL;
    SecKeyRef peer1OctagonEncryptionKey = NULL;
    SOSFullPeerInfoRef fullPeer1WithBackup = SOSCreateFullPeerInfoFromName(CFSTR("peer1WithBackupID"), &peer1SigningKey, &peer1OctagonSigningKey, &peer1OctagonEncryptionKey, &localError);
    ok(fullPeer1WithBackup, "Allocate peer 1 (%@)", localError);
    CFReleaseNull(localError);

    CFDataRef peer1BackupPublic = SOSCopyDeviceBackupPublicKey(entropy1, &localError);
    ok(peer1BackupPublic, "Got backup key (%@)", localError);
    CFReleaseNull(localError);

    SOSFullPeerInfoUpdateBackupKey(fullPeer1WithBackup, peer1BackupPublic, &localError);

    SOSPeerInfoRef peer1WithBackup = SOSFullPeerInfoGetPeerInfo(fullPeer1WithBackup);

    SecKeyRef peer2SigningKey = NULL;
    SecKeyRef peer2OctagonSigningKey = NULL;
    SecKeyRef peer2OctagonEncryptionKey = NULL;
    SOSFullPeerInfoRef fullPeer2WithBackup = SOSCreateFullPeerInfoFromName(CFSTR("peer2WithBackupID"), &peer2SigningKey, &peer2OctagonSigningKey, &peer2OctagonEncryptionKey, &localError);
    ok(fullPeer2WithBackup, "Allocate peer 2 (%@)", localError);
    CFReleaseNull(localError);

    CFDataRef peer2BackupPublic = SOSCopyDeviceBackupPublicKey(entropy2, &localError);
    ok(peer2BackupPublic, "Got backup key (%@)", localError);
    CFReleaseNull(localError);

    SOSFullPeerInfoUpdateBackupKey(fullPeer2WithBackup, peer2BackupPublic, &localError);

    SOSPeerInfoRef peer2WithBackup = SOSFullPeerInfoGetPeerInfo(fullPeer2WithBackup);

    SOSBackupSliceKeyBagRef vb = SOSBackupSliceKeyBagCreate(kCFAllocatorDefault, piSet, &localError);
    ok(vb == NULL, "Should fail with no peers (%@)", localError);
    CFReleaseNull(localError);
    CFReleaseNull(vb);

    CFSetAddValue(piSet, peer1WithBackup);
    CFSetAddValue(piSet, peer2WithBackup);

    SOSBackupSliceKeyBagRef vb2 = NULL;

    vb = SOSBackupSliceKeyBagCreate(kCFAllocatorDefault, piSet, &localError);
    ok(vb != NULL, "Allocation: (%@)", localError);
    CFReleaseNull(localError);

    vb2 = EncodeDecode(vb);

    ok(vb2 != NULL, "transcoded");

    CFReleaseNull(vb);
    CFReleaseNull(vb2);
    CFReleaseNull(piSet);

    CFReleaseNull(peer1SigningKey);
    CFReleaseNull(peer2OctagonSigningKey);
    CFReleaseNull(peer1BackupPublic);
    CFReleaseNull(fullPeer1WithBackup);

    CFReleaseNull(peer2SigningKey);
    CFReleaseNull(peer2OctagonSigningKey);
    CFReleaseNull(peer2BackupPublic);
    CFReleaseNull(fullPeer2WithBackup);

    CFReleaseNull(entropy1);
    CFReleaseNull(entropy2);
}
#endif

int sc_153_backupslicekeybag(int argc, char *const *argv)
{
#if SOS_ENABLED
    plan_tests(12);
    tests();
#else
    plan_tests(0);
#endif
    return 0;
}

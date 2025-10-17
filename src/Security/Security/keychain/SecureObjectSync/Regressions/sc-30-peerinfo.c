/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 13, 2021.
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
#include <Security/SecBase.h>
#include <Security/SecItem.h>
#include <Security/SecKey.h>
#include <Security/SecKeyPriv.h>
#include "keychain/SecureObjectSync/SOSPeerInfoDER.h"

#include "keychain/SecureObjectSync/SOSCircle.h"
#include <Security/SecureObjectSync/SOSPeerInfo.h>
#include "keychain/SecureObjectSync/SOSPeerInfoCollections.h"
#include "keychain/SecureObjectSync/SOSInternal.h"
#include "keychain/SecureObjectSync/SOSUserKeygen.h"

#include <utilities/SecCFWrappers.h>

#include <CoreFoundation/CoreFoundation.h>

#include <stdlib.h>
#include <unistd.h>

#include "SOSCircle_regressions.h"

#include "SOSRegressionUtilities.h"

#if SOS_ENABLED

#if TARGET_OS_IPHONE
#include <MobileGestalt.h>
#endif

static CFDataRef CopyTestBackupKey(void) {
    static uint8_t data[] = { 'A', 'b', 'c' };

    return CFDataCreate(kCFAllocatorDefault, data, sizeof(data));
}

static bool PeerInfoRoundTrip(SOSPeerInfoRef pi) {
    bool retval = false;
    size_t size = SOSPeerInfoGetDEREncodedSize(pi, NULL);
    uint8_t buffer[size];
    const uint8_t *buffer_p = SOSPeerInfoEncodeToDER(pi, NULL, buffer, buffer + sizeof(buffer));
    ok(buffer_p != NULL, "encode");
    if(buffer_p == NULL) return false;
    SOSPeerInfoRef pi2 = SOSPeerInfoCreateFromDER(NULL, NULL, &buffer_p, buffer + sizeof(buffer));
    ok(pi2 != NULL, "decode");
    if(!pi2) return false;
    ok(CFEqual(pi, pi2), "Decode matches");
    if(CFEqual(pi, pi2)) retval = true;
    CFReleaseNull(pi2);
    return retval;
}

static bool FullPeerInfoRoundTrip(SOSFullPeerInfoRef fpi) {
    bool retval = false;
    size_t size = SOSFullPeerInfoGetDEREncodedSize(fpi, NULL);
    uint8_t buffer[size];
    const uint8_t *buffer_p = SOSFullPeerInfoEncodeToDER(fpi, NULL, buffer, buffer + sizeof(buffer));
    ok(buffer_p != NULL, "encode");
    if(buffer_p == NULL) return false;
    SOSFullPeerInfoRef fpi2 = SOSFullPeerInfoCreateFromDER(NULL, NULL, &buffer_p, buffer + sizeof(buffer));
    ok(fpi2 != NULL, "decode");
    if(!fpi2) return false;
    ok(CFEqual(fpi, fpi2), "Decode matches");
    if(CFEqual(fpi, fpi2)) retval = true;
    CFReleaseNull(fpi2);
    return retval;
}

static int kTestTestCount = 24;
static void tests(void)
{
    SecKeyRef signingKey = NULL;
    SecKeyRef octagonSigningKey = NULL;
    SecKeyRef octagonEncryptionKey = NULL;
    SOSFullPeerInfoRef fpi = SOSCreateFullPeerInfoFromName(CFSTR("Test Peer"), &signingKey, &octagonSigningKey, &octagonEncryptionKey, NULL);
    SOSPeerInfoRef pi = SOSFullPeerInfoGetPeerInfo(fpi);

    ok(NULL != pi, "info creation");
    
    ok(PeerInfoRoundTrip(pi), "PeerInfo safely round-trips");
    ok(FullPeerInfoRoundTrip(fpi), "FullPeerInfo safely round-trips");

    // Application ticket time.
    CFDataRef cfpassword = CFDataCreate(NULL, (uint8_t *) "FooFooFoo", 10);
    CFErrorRef error = NULL;

    CFDataRef parameters = SOSUserKeyCreateGenerateParameters(&error);
    ok(parameters, "No parameters!");
    ok(error == NULL, "Error: (%@)", error);
    CFReleaseNull(error);

    SecKeyRef user_privkey = SOSUserKeygen(cfpassword, parameters, &error);
    CFReleaseSafe(cfpassword);
    CFReleaseNull(parameters);
    SecKeyRef user_pubkey = SecKeyCreatePublicFromPrivate(user_privkey);

    ok(SOSFullPeerInfoPromoteToApplication(fpi, user_privkey, &error), "Promote to Application");
    ok(SOSPeerInfoApplicationVerify(SOSFullPeerInfoGetPeerInfo(fpi), user_pubkey, &error), "Promote to Application");
    
    pi = SOSFullPeerInfoGetPeerInfo(fpi);
    ok(PeerInfoRoundTrip(pi), "PeerInfo safely round-trips");

    CFDataRef testBackupKey = CopyTestBackupKey();

    ok(SOSFullPeerInfoUpdateBackupKey(fpi, testBackupKey, &error), "Set Backup (%@)", error);
    CFReleaseNull(error);

    CFReleaseNull(testBackupKey); // Make sure our ref doesn't save them.
    testBackupKey = CopyTestBackupKey();

    pi = SOSFullPeerInfoGetPeerInfo(fpi);
    CFDataRef piBackupKey = SOSPeerInfoCopyBackupKey(pi);

    ok(CFEqualSafe(testBackupKey, piBackupKey), "Same Backup Key");

    ok(PeerInfoRoundTrip(pi), "PeerInfo safely round-trips with backup key");

    CFReleaseNull(piBackupKey);
    piBackupKey = SOSPeerInfoCopyBackupKey(pi);
    ok(CFEqualSafe(testBackupKey, piBackupKey), "Same Backup Key after round trip");

    // Don't own the piBackupKey key
    CFReleaseNull(testBackupKey);
    CFReleaseNull(piBackupKey);
    CFReleaseNull(user_privkey);
    CFReleaseNull(user_pubkey);

    CFReleaseNull(signingKey);
    CFReleaseNull(octagonSigningKey);
    CFReleaseNull(octagonEncryptionKey);
    CFReleaseNull(fpi);
}
#endif

int sc_30_peerinfo(int argc, char *const *argv)
{
#if SOS_ENABLED
    plan_tests(kTestTestCount);
    tests();
#else
    plan_tests(0);
#endif
	return 0;
}

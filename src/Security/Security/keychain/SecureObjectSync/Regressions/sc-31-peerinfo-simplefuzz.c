/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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

static unsigned long kTestCount = 2;
static unsigned long kTestFuzzerCount = 20000;

static void tests(void)
{
    SecKeyRef signingKey = NULL;
    SecKeyRef octagonSigningKey = NULL;
    SecKeyRef octagonEncryptionKey = NULL;
    SOSFullPeerInfoRef fpi = SOSCreateFullPeerInfoFromName(CFSTR("Test Peer"), &signingKey, &octagonSigningKey, &octagonEncryptionKey, NULL);
    SOSPeerInfoRef pi = SOSFullPeerInfoGetPeerInfo(fpi);
    unsigned long count;
    
    ok(NULL != pi, "info creation");
    size_t size = SOSPeerInfoGetDEREncodedSize(pi, NULL);

    uint8_t buffer[size+100]; // make the buffer long enough to hold the DER + some room for the fuzzing
    
    const uint8_t *buffer_p = SOSPeerInfoEncodeToDER(pi, NULL, buffer, buffer + sizeof(buffer));
    
    ok(buffer_p != NULL, "encode");

    size_t length = (buffer + sizeof(buffer)) - buffer_p;
    // diag("size %lu length %lu\n", size, length);
    uint8_t buffer2[length];
    if(buffer_p == NULL) goto errOut;

        for (count = 0; count < kTestFuzzerCount; count++) {
            memcpy(buffer2, buffer_p, length);

            const uint8_t *startp = buffer2;
            size_t offset = arc4random_uniform((u_int32_t)length);
            uint8_t value = arc4random() & 0xff;
            // diag("Offset %lu value %d\n", offset, value);
            buffer2[offset] = value;

            SOSPeerInfoRef pi2 = SOSPeerInfoCreateFromDER(NULL, NULL, &startp, buffer2 + length);
            CFReleaseNull(pi2);
            ok(1, "fuzz");
        }
    
errOut:
    CFReleaseNull(signingKey);
    CFReleaseNull(octagonSigningKey);
    CFReleaseNull(octagonEncryptionKey);
    CFReleaseNull(fpi);
}
#endif

int sc_31_peerinfo(int argc, char *const *argv)
{
#if SOS_ENABLED
    plan_tests((int)(kTestCount + kTestFuzzerCount));
    tests();
#else
    plan_tests(0);
#endif
	return 0;
}

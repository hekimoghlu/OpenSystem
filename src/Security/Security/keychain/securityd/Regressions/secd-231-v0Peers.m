/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 28, 2025.
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

#include <CoreFoundation/CFDictionary.h>

#include "keychain/SecureObjectSync/SOSAccount.h"
#include <Security/SecureObjectSync/SOSCloudCircle.h>
#include "keychain/SecureObjectSync/SOSInternal.h"
#include "keychain/SecureObjectSync/SOSUserKeygen.h"
#include "keychain/SecureObjectSync/SOSTransport.h"
#include "keychain/SecureObjectSync/SOSAccountTrustClassic+Circle.h"

#include <stdlib.h>
#include <unistd.h>

#include "secd_regressions.h"
#include "SOSTestDataSource.h"

#include "SOSRegressionUtilities.h"
#include <utilities/SecCFWrappers.h>
#include <Security/SecKeyPriv.h>

#include "keychain/securityd/SOSCloudCircleServer.h"

#include "SOSAccountTesting.h"

#include "SecdTestKeychainUtilities.h"
#if SOS_ENABLED

static void tests(void)
{
    CFErrorRef error = NULL;
    CFDataRef cfpassword = CFDataCreate(NULL, (uint8_t *) "FooFooFoo", 10);
    CFStringRef cfaccount = CFSTR("test@test.org");

    CFMutableDictionaryRef changes = CFDictionaryCreateMutableForCFTypes(kCFAllocatorDefault);

    SOSAccount* alice_account = CreateAccountForLocalChanges( CFSTR("Alice"), CFSTR("TestSource"));
        
    is(ProcessChangesUntilNoChange(changes, alice_account, NULL), 1, "updates");

    ok(SOSAccountAssertUserCredentialsAndUpdate(alice_account, cfaccount, cfpassword, &error), "Credential setting (%@)", error);
    CFReleaseNull(error);
    
    ok(SOSAccountResetToOffering_wTxn(alice_account, &error), "Reset to offering (%@)", error);
    CFReleaseNull(error);
    
    is(ProcessChangesUntilNoChange(changes, alice_account, NULL), 1, "updates");
    
    // this is going to construct a V0 peer, and have alice plant it into the circle.
    SecKeyRef correctUserKey = alice_account.accountPrivateKey;
    SOSFullPeerInfoRef bobsV0FPI = SOSTestV0FullPeerInfo(CFSTR("Bob"), correctUserKey, CFSTR("12A365"), SOSPeerInfo_iOS);
    SOSCircleRef aliceAccountCircle = alice_account.trust.trustedCircle;
    SOSCircleRequestAdmission(aliceAccountCircle, correctUserKey, bobsV0FPI, &error);

    {
        CFArrayRef applicants = SOSAccountCopyApplicants(alice_account, &error);
        
        ok(applicants && CFArrayGetCount(applicants) == 1, "See one applicant %@ (%@)", applicants, error);
        ok(SOSAccountAcceptApplicants(alice_account, applicants, &error), "Alice accepts (%@)", error);
        CFReleaseNull(error);
        CFReleaseNull(applicants);
    }
    
    is(ProcessChangesUntilNoChange(changes, alice_account, NULL), 1, "updates");
     
    CFArrayRef peers = SOSAccountCopyViewUnawarePeers_wTxn(alice_account, &error);
    ok(peers, "peers should not be nil");
    is(CFArrayGetCount(peers), 1, "count should be 1");

    CFReleaseNull(cfpassword);

    // now change the account key on alice account
    cfpassword = CFDataCreate(NULL, (uint8_t *) "NewFooFooFoo", 10);

    CFDataRef parameters = SOSUserKeyCreateGenerateParameters(&error);
    SecKeyRef user_private = SOSUserKeygen(cfpassword, parameters, &error);
    SOSAccountSetParameters(alice_account, parameters);
    SecKeyRef publicKey = SecKeyCreatePublicFromPrivate(user_private);
    alice_account.accountKey = publicKey;

    // now there should be 0 active view unaware peers
    CFArrayRef activeViewUnawarePeers = SOSAccountCopyViewUnawarePeers_wTxn(alice_account, &error);
    ok(activeViewUnawarePeers, "peers should not be nil");
    is(CFArrayGetCount(activeViewUnawarePeers), 0, "count should be 0");

    // assert new password
    ok(SOSAccountAssertUserCredentialsAndUpdate(alice_account, cfaccount, cfpassword, &error), "Credential setting (%@)", error);
    CFReleaseNull(error);
    is(ProcessChangesUntilNoChange(changes, alice_account, NULL), 1, "updates");

    CFReleaseNull(parameters);
    CFReleaseNull(user_private);
    CFReleaseNull(publicKey);

    if(CFArrayGetCount(peers) > 0) {
        SOSPeerInfoRef bobPeer = (SOSPeerInfoRef)CFArrayGetValueAtIndex(peers, 0);
        ok(CFStringCompare(SOSPeerInfoGetPeerID(bobPeer), SOSPeerInfoGetPeerID(SOSFullPeerInfoGetPeerInfo(bobsV0FPI)), 0) == kCFCompareEqualTo, "PeerInfo's should be equal");
    
        [alice_account removeV0Peers:^(bool removedV0Peer, NSError *error) {
            is(removedV0Peer, true, "v0 peer should be removed");
        }];
    
        is(ProcessChangesUntilNoChange(changes, alice_account, NULL), 1, "updates");
    }

    is(SOSCircleCountPeers(alice_account.trust.trustedCircle), 1, "Bob should not be in the circle");

    CFReleaseNull(changes);
    CFReleaseNull(error);
    CFReleaseNull(cfpassword);
    CFReleaseNull(bobsV0FPI);

    alice_account = nil;
    SOSTestCleanup();
}
#endif

int secd_231_v0Peers(int argc, char *const *argv)
{
#if SOS_ENABLED
    plan_tests(41);
    enableSOSCompatibilityForTests();

    secd_test_setup_temp_keychain(__FUNCTION__, NULL);
    tests();
    secd_test_teardown_delete_temp_keychain(__FUNCTION__);
#else
    plan_tests(0);
#endif
    return 0;
}

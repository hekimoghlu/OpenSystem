/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 9, 2021.
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

//
//  secd-68-fullPeerInfoIntegrity.m
//  secdRegressions
//
//  Created by Richard Murphy on 4/30/20.
//

#import <Foundation/Foundation.h>

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

static NSString *makeCircle(SOSAccount* testaccount, CFMutableDictionaryRef changes) {
    CFErrorRef error = NULL;

    // Every time we resetToOffering should result in a new fpi
    NSString *lastPeerID = testaccount.peerID;
    ok(SOSAccountResetToOffering_wTxn(testaccount, &error), "Reset to offering (%@)", error);
    CFReleaseNull(error);
    is(ProcessChangesUntilNoChange(changes, testaccount, NULL), 1, "updates");
    NSString *currentPeerID = testaccount.peerID;
    ok(![lastPeerID isEqualToString:currentPeerID], "peerID changed on circle reset");
    return currentPeerID;
}

static void tests(void) {
    CFErrorRef error = NULL;
    CFDataRef cfpassword = CFDataCreate(NULL, (uint8_t *) "FooFooFoo", 10);
    CFStringRef cfaccount = CFSTR("test@test.org");

    CFMutableDictionaryRef changes = CFDictionaryCreateMutableForCFTypes(kCFAllocatorDefault);
    SOSAccount* testaccount = CreateAccountForLocalChanges(CFSTR("TestDev"), CFSTR("TestSource"));

    // Just making an account object to mess with
    ok(SOSAccountAssertUserCredentialsAndUpdate(testaccount, cfaccount, cfpassword, &error), "Credential setting (%@)", error);
    is(ProcessChangesUntilNoChange(changes, testaccount, NULL), 1, "updates");
    CFReleaseNull(error);

    // make a circle then make sure fpi isn't reset just for a normal ensureFullPeerAvailable
    NSString *lastPeerID = makeCircle(testaccount, changes);
    ok([testaccount.trust ensureFullPeerAvailable:testaccount err:&error], "fullPeer is available");
    NSString *currentPeerID = testaccount.peerID;
    ok([lastPeerID isEqualToString: currentPeerID], "peerID did not alter in trip through ensureFullPeerAvailable");


    // leaving a circle should reset the fpi
    lastPeerID = makeCircle(testaccount, changes);
    ok([testaccount.trust leaveCircle:testaccount err:&error], "leave the circle %@", error);
    CFReleaseNull(error);
    is(ProcessChangesUntilNoChange(changes, testaccount, NULL), 1, "updates");
    currentPeerID = testaccount.peerID;
    ok(![lastPeerID isEqualToString:currentPeerID], "peerID changed on leaving circle");

    // break the fullpeerinfo by purging the private key - then fix in ensureFullPeerAvailable
    lastPeerID = makeCircle(testaccount, changes);
    ok(SOSFullPeerInfoPurgePersistentKey(testaccount.fullPeerInfo, &error), "purging persistent key %@", error);
    currentPeerID = testaccount.peerID;
    ok([lastPeerID isEqualToString:currentPeerID], "pre-ensuring peerID remains the same");
    lastPeerID = currentPeerID;
    ok([testaccount.trust ensureFullPeerAvailable:testaccount err:&error], "fullPeer is available");
    currentPeerID = testaccount.peerID;
    ok(![lastPeerID isEqualToString: currentPeerID], "peerID changed because fullPeerInfo fixed in ensureFullPeerAvailable");
    lastPeerID = currentPeerID;

    // If that last thing worked this peer won't be in the circle any more - changing fpi changes "me"
    ok(SOSAccountAssertUserCredentialsAndUpdate(testaccount, cfaccount, cfpassword, &error), "Credential setting (%@)", error);
    is(ProcessChangesUntilNoChange(changes, testaccount, NULL), 1, "updates");
    CFReleaseNull(error);
    ok(![testaccount isInCircle: &error], "No longer in circle");

    // This join should work because the peer we left in the circle will be a ghost and there are no other peers
    ok(SOSAccountJoinCircles_wTxn(testaccount, &error), "Apply to circle (%@)", error);
    CFReleaseNull(error);
    is(ProcessChangesUntilNoChange(changes, testaccount, NULL), 1, "updates");
    ok([testaccount isInCircle: &error], "Is in circle");
    currentPeerID = testaccount.peerID;
    ok(![lastPeerID isEqualToString: currentPeerID], "peerID changed because fullPeerInfo changed during join");

    CFReleaseNull(cfpassword);
    testaccount = nil;
    SOSTestCleanup();
}
#endif

int secd_68_fullPeerInfoIntegrity(int argc, char *const *argv)
{
#if SOS_ENABLED
    plan_tests(29);
    enableSOSCompatibilityForTests();
    secd_test_setup_temp_keychain(__FUNCTION__, NULL);
    tests();
    secd_test_teardown_delete_temp_keychain(__FUNCTION__);
#else
    plan_tests(0);
#endif
    return 0;
}

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 5, 2025.
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


typedef void (^stir_block)(int expected_iterations);
typedef int (^execute_block)(void);

static void stirBetween(stir_block stir, ...) {
    va_list va;
    va_start(va, stir);

    execute_block execute = NULL;

    while ((execute = va_arg(va, execute_block)) != NULL)
        stir(execute());
}

static void VerifyCountAndAcceptAllApplicants(SOSAccount* account, int expected)
{
    CFErrorRef error = NULL;
    CFArrayRef applicants = SOSAccountCopyApplicants(account, &error);

    SKIP: {
        skip("Empty applicant array", 2, applicants);

        is(CFArrayGetCount(applicants), expected, "Applicants: %@ (%@)", applicants, error);
        CFReleaseNull(error);

        ok(SOSAccountAcceptApplicants(account , applicants, &error), "Accepting all (%@)", error);
        CFReleaseNull(error);
    }

    CFReleaseNull(applicants);
}


static void tests(void)
{
    __block CFErrorRef error = NULL;
    CFDataRef cfpassword = CFDataCreate(NULL, (uint8_t *) "FooFooFoo", 10);
    CFStringRef cfaccount = CFSTR("test@test.org");

    CFMutableDictionaryRef changes = CFDictionaryCreateMutableForCFTypes(kCFAllocatorDefault);

    const CFStringRef data_source_name = CFSTR("TestSource");

    SOSAccount* alice_account = CreateAccountForLocalChanges(CFSTR("Alice"), data_source_name);
    SOSAccount* bob_account = CreateAccountForLocalChanges(CFSTR("Bob"), data_source_name);
    SOSAccount* carole_account = CreateAccountForLocalChanges(CFSTR("Carole"), data_source_name);

    SOSAccount* alice_resurrected = NULL;

    __block CFDataRef frozen_alice = NULL;


    stirBetween(^(int expected){
        is(ProcessChangesUntilNoChange(changes, alice_account, bob_account, carole_account, NULL), expected, "stirring");
    }, ^{
        ok(SOSAccountAssertUserCredentialsAndUpdate(bob_account , cfaccount, cfpassword, &error), "bob credential setting (%@)", error);

        return 1;
    }, ^{
        ok(SOSAccountAssertUserCredentialsAndUpdate(alice_account , cfaccount, cfpassword, &error), "alice credential setting (%@)", error);
        CFReleaseNull(error);

        ok(SOSAccountAssertUserCredentialsAndUpdate(carole_account,  cfaccount, cfpassword, &error), "carole credential setting (%@)", error);
        CFReleaseNull(error);

        ok(SOSAccountResetToOffering_wTxn(alice_account , &error), "Reset to offering (%@)", error);
        CFReleaseNull(error);

        return 2;
    }, ^{
        ok(SOSAccountJoinCircles_wTxn(bob_account , &error), "Bob Applies (%@)", error);
        CFReleaseNull(error);

        return 2;
    }, ^{
        VerifyCountAndAcceptAllApplicants(alice_account, 1);

        return 3;
    }, ^{
        accounts_agree("bob&alice pair", bob_account, alice_account);
        is(SOSAccountGetLastDepartureReason(bob_account, &error), kSOSNeverLeftCircle, "Bob affirms he hasn't left.");

        CFArrayRef peers = SOSAccountCopyPeers(alice_account, &error);
        ok(peers && CFArrayGetCount(peers) == 2, "See two peers %@ (%@)", peers, error);
        CFReleaseNull(peers);

        return 1;
    }, ^{

        NSError *ns_error = nil;
        frozen_alice = (CFDataRef) CFBridgingRetain([alice_account encodedData:&ns_error]);
        ok(frozen_alice, "Copy encoded %@", ns_error);
        ns_error = nil;
        
        SOSAccountPurgePrivateCredential(alice_account);

        ok([alice_account.trust leaveCircle:alice_account err:&error], "Alice Leaves (%@)", error);
        CFReleaseNull(error);

        return 2;
    }, ^{

        accounts_agree("Alice bails", bob_account, alice_account);

        {
            CFArrayRef concurring = SOSAccountCopyConcurringPeers(alice_account, &error);

            ok(concurring && CFArrayGetCount(concurring) == 2, "See two concurring %@ (%@)", concurring, error);
            CFReleaseNull(error);
            CFReleaseNull(concurring);
        }

        return 1;
    },
    NULL);

    alice_resurrected = CreateAccountForLocalChangesFromData(frozen_alice, CFSTR("Alice risen"), data_source_name);
    // This is necessary from the change that makes accounts not inflate if the private key was lost - alice_resurected now
    // Starts as a brand new account, so this whole series of tests needs to amount to "is this brand new"?
    // The trigger is alice leaving the circle - that kills the deviceKey.
    ProcessChangesUntilNoChange(changes, alice_resurrected, bob_account, carole_account, NULL);

    stirBetween(^(int expected){
        is(ProcessChangesUntilNoChange(changes, alice_resurrected, bob_account, carole_account, NULL), expected, "stirring");
    }, ^{
        ok(SOSAccountAssertUserCredentialsAndUpdate(alice_resurrected,  cfaccount, cfpassword, &error), "alice_resurrected credential setting (%@)", error);
        CFReleaseNull(error);
        return 1;
    }, ^{
        ok(![alice_resurrected isInCircle:&error], "Ressurrected not in circle: %@", error);
        CFReleaseNull(error);

        ok(SOSAccountJoinCircles_wTxn(alice_resurrected, &error), "Risen-alice Applies (%@)", error);
        CFReleaseNull(error);
        return 2;
    }, ^{
        VerifyCountAndAcceptAllApplicants(bob_account, 1);
        return 3;
    },
    NULL);

    CFReleaseNull(frozen_alice);
    alice_account = nil;
    bob_account = nil;
    SOSTestCleanup();
}
#endif

int secd_63_account_resurrection(int argc, char *const *argv)
{
#if SOS_ENABLED
    plan_tests(73);
    enableSOSCompatibilityForTests();
    secd_test_setup_temp_keychain(__FUNCTION__, NULL);
    tests();
    secd_test_teardown_delete_temp_keychain(__FUNCTION__);
#else
    plan_tests(0);
#endif
    return 0;
}

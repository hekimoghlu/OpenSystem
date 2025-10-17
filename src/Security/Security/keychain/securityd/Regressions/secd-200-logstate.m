/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 20, 2022.
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
//  secd-200-logstate.c
//  sec
//

#include <stdio.h>




#include <Security/SecBase.h>
#include <Security/SecItem.h>

#include <CoreFoundation/CFDictionary.h>

#include "keychain/SecureObjectSync/SOSAccount.h"
#include <Security/SecureObjectSync/SOSCloudCircle.h>
#include "keychain/SecureObjectSync/SOSInternal.h"
#include "keychain/SecureObjectSync/SOSUserKeygen.h"
#include "keychain/SecureObjectSync/SOSTransport.h"

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

#define HOW_MANY_MINIONS 4

static bool SOSArrayForEachAccount(CFArrayRef accounts, bool (^operation)(SOSAccount* account)) {
    __block bool retval = true;
    CFArrayForEach(accounts, ^(const void *value) {
        SOSAccount* account = (__bridge SOSAccount*) value;
        retval &= operation(account);
    });
    return retval;
}


static inline void FeedChangesToMasterMinions(CFMutableDictionaryRef changes, SOSAccount* master_account, CFArrayRef minion_accounts) {
    FeedChangesTo(changes, master_account);
    SOSArrayForEachAccount(minion_accounts, ^bool(SOSAccount* account) {
        FeedChangesTo(changes, account);
        return true;
    });
    FeedChangesTo(changes, master_account);

}


static inline bool ProcessChangesOnceMasterMinions(CFMutableDictionaryRef changes, SOSAccount* master_account, CFArrayRef minion_accounts) {
    bool result = FillAllChanges(changes);
    FeedChangesToMasterMinions(changes, master_account, minion_accounts);
    return result;
}

static inline int ProcessChangesForMasterAndMinions(CFMutableDictionaryRef changes, SOSAccount* master_account, CFArrayRef minion_accounts) {
    int result = 0;
    bool new_data = false;
    do {
        new_data = ProcessChangesOnceMasterMinions(changes, master_account, minion_accounts);
        ++result;
    } while (new_data);
    return result;
}

static bool MakeTheBigCircle(CFMutableDictionaryRef changes, SOSAccount* master_account, CFArrayRef minion_accounts, CFErrorRef *error) {
    bool retval = SOSAccountResetToOffering_wTxn(master_account, error);
    if(!retval)
        return retval;
    ProcessChangesForMasterAndMinions(changes, master_account, minion_accounts);
    retval = SOSArrayForEachAccount(minion_accounts, ^bool(SOSAccount* account) {
        bool localret = SOSAccountJoinCircles_wTxn(account, error);
        ProcessChangesForMasterAndMinions(changes, master_account, minion_accounts);
        return localret;
    });
    require_quiet(retval, errOut);
    CFArrayRef applicants = SOSAccountCopyApplicants(master_account, error);
    retval = SOSAccountAcceptApplicants(master_account , applicants, error);
    CFReleaseNull(applicants);
    ProcessChangesForMasterAndMinions(changes, master_account, minion_accounts);
errOut:
    return retval;
}


static CFArrayRef CreateManyAccountsForLocalChanges(CFStringRef namefmt, CFStringRef data_source_name, size_t howmany)
    CF_FORMAT_FUNCTION(1, 0);

static CFArrayRef CreateManyAccountsForLocalChanges(CFStringRef name, CFStringRef data_source_name, size_t howmany) {
    CFMutableArrayRef accounts = CFArrayCreateMutable(kCFAllocatorDefault, howmany, &kCFTypeArrayCallBacks);
    
    for(size_t i = 0; i < howmany; i++) {
        CFStringRef tmpname = CFStringCreateWithFormat(kCFAllocatorDefault, NULL, CFSTR("%@%ld"), name, (long)i);
        SOSAccount* tmp = CreateAccountForLocalChanges(tmpname, CFSTR("TestSource"));
        CFArraySetValueAtIndex(accounts, i, (__bridge const void *)(tmp));
        CFReleaseNull(tmpname);
    }
    return accounts;
}

static bool AssertAllCredentialsAndUpdate(CFMutableDictionaryRef changes, SOSAccount* master_account, CFArrayRef minion_accounts, CFStringRef user_account, CFDataRef user_password, CFErrorRef *error) {
    __block bool retval = SOSAccountAssertUserCredentialsAndUpdate(master_account, user_account, user_password, error);
    ProcessChangesForMasterAndMinions(changes, master_account, minion_accounts);
    retval &= SOSArrayForEachAccount(minion_accounts, ^bool(SOSAccount* account) {
        CFReleaseNull(*error);
        return SOSAccountAssertUserCredentialsAndUpdate(account, user_account, user_password, error);
    });
    CFReleaseNull(*error);

    return retval;
}

static void timeval_delta(struct timeval *delta, struct timeval *start, struct timeval *end) {
    if(end->tv_usec >= start->tv_usec) {
        delta->tv_usec = end->tv_usec - start->tv_usec;
    } else {
        end->tv_sec--;
        end->tv_usec += 1000000;
        delta->tv_usec = end->tv_usec - start->tv_usec;
    }
    delta->tv_sec = end->tv_sec - start->tv_sec;
}

static void reportTime(int peers, void(^action)(void)) {
    struct rusage start_rusage;
    struct rusage end_rusage;
    struct timeval delta_utime;
    struct timeval delta_stime;

    getrusage(RUSAGE_SELF, &start_rusage);
    action();
    getrusage(RUSAGE_SELF, &end_rusage);
    timeval_delta(&delta_utime, &start_rusage.ru_utime,  &end_rusage.ru_utime);
    timeval_delta(&delta_stime, &start_rusage.ru_stime, &end_rusage.ru_stime);

    diag("AccountLogState for %d peers: %ld.%06d user  %ld.%06d system", peers,
         delta_utime.tv_sec, delta_utime.tv_usec,
         delta_stime.tv_sec, delta_stime.tv_usec);
}

static void tests(void)
{
    NSError* ns_error = nil;
    CFErrorRef error = NULL;
    CFDataRef cfpassword = CFDataCreate(NULL, (uint8_t *) "FooFooFoo", 10);
    CFStringRef cfaccount = CFSTR("test@test.org");
    
    CFMutableDictionaryRef changes = CFDictionaryCreateMutableForCFTypes(kCFAllocatorDefault);
    
    SOSAccount* master_account = CreateAccountForLocalChanges(CFSTR("master"), CFSTR("TestSource"));
    CFArrayRef minion_accounts = CreateManyAccountsForLocalChanges(CFSTR("minion"), CFSTR("TestSource"), HOW_MANY_MINIONS);
    
    ok(AssertAllCredentialsAndUpdate(changes, master_account, minion_accounts, cfaccount, cfpassword, &error), "Credential setting (%@)", error);
    CFReleaseNull(cfpassword);

    secLogEnable();
    reportTime(1, ^{
        SOSAccountLogState(master_account);
    });
    secLogDisable();

    ok(MakeTheBigCircle(changes, master_account, minion_accounts, &error), "Get Everyone into the circle %@", error);
    
    diag("WHAT?");
    secLogEnable();
    reportTime(HOW_MANY_MINIONS+1, ^{
        SOSAccountLogState(master_account);
    });
    SOSAccountLogViewState(master_account);
    secLogDisable();
    
    NSData* acctData = [master_account encodedData:&ns_error];
    diag("Account DER Size is %lu for %d peers", (unsigned long)[acctData length], HOW_MANY_MINIONS+1);
    ns_error = nil;

    SOSAccountTrustClassic* trust = master_account.trust;
    CFDataRef circleData = SOSCircleCopyEncodedData(trust.trustedCircle, kCFAllocatorDefault, &error);
    diag("Circle DER Size is %ld for %d peers", CFDataGetLength(circleData), HOW_MANY_MINIONS+1);
    CFReleaseNull(circleData);
    CFReleaseNull(error);

    CFDataRef peerData = SOSPeerInfoCopyEncodedData(master_account.peerInfo, kCFAllocatorDefault, &error);
    diag("Peer DER Size is %ld", CFDataGetLength(peerData));
    CFReleaseNull(peerData);
    CFReleaseNull(error);

    CFReleaseNull(error);
    CFReleaseNull(minion_accounts);
    
    SOSTestCleanup();
    
}
#endif

int secd_200_logstate(int argc, char *const *argv)
{
#if SOS_ENABLED
    plan_tests(((HOW_MANY_MINIONS+1)*10 + 1));
    enableSOSCompatibilityForTests();
    secd_test_setup_temp_keychain(__FUNCTION__, NULL);
    tests();
    secd_test_teardown_delete_temp_keychain(__FUNCTION__);
#else
    plan_tests(0);
#endif
    return 0;
}

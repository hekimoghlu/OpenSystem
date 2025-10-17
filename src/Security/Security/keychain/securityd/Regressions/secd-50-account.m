/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 14, 2024.
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

#include "keychain/SecureObjectSync/SOSAccount.h"
#include <Security/SecureObjectSync/SOSCloudCircle.h>
#include "keychain/SecureObjectSync/SOSInternal.h"
#include "keychain/SecureObjectSync/SOSUserKeygen.h"
#include "keychain/SecureObjectSync/SOSAccountTrustClassic.h"
#include "keychain/SecureObjectSync/SOSAccountTrustClassic+Circle.h"
#include <stdlib.h>
#include <unistd.h>

#include "secd_regressions.h"
#include "SOSTestDataSource.h"

#include "SOSRegressionUtilities.h"
#include <utilities/SecCFWrappers.h>

#include "keychain/securityd/SOSCloudCircleServer.h"
#include "SecdTestKeychainUtilities.h"
#include "SOSAccountTesting.h"

#if SOS_ENABLED

static int kTestTestCount = 9 + kSecdTestSetupTestCount;
static void tests(void)
{
    NSError* error = nil;
    CFErrorRef cfError = NULL;
    CFDataRef cfpassword = CFDataCreate(NULL, (uint8_t *) "FooFooFoo", 10);
    CFStringRef cfaccount = CFSTR("test@test.org");

    SOSDataSourceFactoryRef test_factory = SOSTestDataSourceFactoryCreate();
    SOSDataSourceRef test_source = SOSTestDataSourceCreate();
    SOSTestDataSourceFactorySetDataSource(test_factory, CFSTR("TestType"), test_source);
    
    SOSAccount* account = CreateAccountForLocalChanges(CFSTR("Test Device"),CFSTR("TestType") );
    
    ok(SOSAccountAssertUserCredentialsAndUpdate(account, cfaccount, cfpassword, &cfError), "Credential setting (%@)", cfError);
    CFReleaseNull(cfError);
    CFReleaseNull(cfpassword);
    
    ok(NULL != account, "Created");

    size_t size = [account.trust getDEREncodedSize:account err:&error];

    error = nil;
    uint8_t buffer[size];

    uint8_t* start = [account.trust encodeToDER:account err:&error start:buffer end:buffer + sizeof(buffer)];
    error = nil;
    
    ok(start, "successful encoding");
    ok(start == buffer, "Used whole buffer");
    
    const uint8_t *der = buffer;
    SOSAccount* inflated = [SOSAccount accountFromDER:&der end:buffer + sizeof(buffer)
                                              factory:test_factory error:&error];

    ok(inflated, "inflated %@", error);
    ok([inflated isEqual:account], "Compares");

    error = nil;

    CFDictionaryRef new_gestalt = SOSCreatePeerGestaltFromName(CFSTR("New Device"));
    ok(SOSAccountResetToOffering_wTxn(account, &cfError), "Reset to Offering  (%@)", error);
    CFReleaseNull(cfError);
    
    is([account getCircleStatus:&cfError], kSOSCCInCircle, "Was in Circle  (%@)", error);
    CFReleaseNull(cfError);
    
    [account.trust updateGestalt:account newGestalt:new_gestalt];
    is([account getCircleStatus:&cfError], kSOSCCInCircle, "Still in Circle  (%@)", error);
    CFReleaseNull(cfError);
    
    SecKeyRef userKey = SOSAccountCopyStashedUserPrivateKey(account, &cfError);
    ok(userKey, "retrieved userKey");
    CFReleaseNull(userKey);
    
    SecKeyRef deviceKey = SOSAccountCopyDevicePrivateKey(account, &cfError);
    ok(deviceKey, "retrieved deviceKey");
    CFReleaseNull(deviceKey);
    
    
    CFReleaseNull(new_gestalt);

    SOSDataSourceFactoryRelease(test_factory);
    SOSDataSourceRelease(test_source, NULL);

    account = nil;
    inflated = nil;
    
    SOSTestCleanup();
}

#endif

int secd_50_account(int argc, char *const *argv)
{
#if SOS_ENABLED
    plan_tests(kTestTestCount);
    secd_test_setup_temp_keychain(__FUNCTION__, NULL);
    tests();
    secd_test_teardown_delete_temp_keychain(__FUNCTION__);
#else
    plan_tests(0);
#endif
    return 0;
}

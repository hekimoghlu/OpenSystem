/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 17, 2023.
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
//  secd-202-recoverykey.c
//  sec
//

#import <Security/Security.h>
#import <Security/SecKeyPriv.h>

#import <Foundation/Foundation.h>

#import <Security/SecRecoveryKey.h>

#import <stdio.h>
#import <stdlib.h>
#import <unistd.h>

#import "secd_regressions.h"
#import "SOSTestDataSource.h"
#import "SOSTestDevice.h"

#import "SOSRegressionUtilities.h"
#import <utilities/SecCFWrappers.h>

#import "SecdTestKeychainUtilities.h"
#include "SOSAccountTesting.h"

#if SOS_ENABLED


const int kTestRecoveryKeyCount = 3;

static void testRecoveryKey(void)
{
    SecRecoveryKey *recoveryKey = NULL;

    recoveryKey = SecRKCreateRecoveryKeyWithError(@"AAAA-AAAA-AAAA-AAAA-AAAA-AAAA-AAGW", NULL);
    ok(recoveryKey, "got recovery key");

    NSData *publicKey = SecRKCopyBackupPublicKey(recoveryKey);
    ok(publicKey, "got publicKey");
}

const int kTestRecoveryKeyBasicNumberIterations = 100;
const int kTestRecoveryKeyBasicCount = 1 * kTestRecoveryKeyBasicNumberIterations;

static void testRecoveryKeyBasic(void)
{
    NSString *recoveryKey = NULL;
    NSError *error = NULL;
    int n;

    for (n = 0; n < kTestRecoveryKeyBasicNumberIterations; n++) {
        recoveryKey = SecRKCreateRecoveryKeyString(&error);
        ok(recoveryKey, "SecRKCreateRecoveryKeyString: %@", error);
    }
}

#endif

int secd_202_recoverykey(int argc, char *const *argv)
{
#if SOS_ENABLED
    plan_tests(kTestRecoveryKeyCount + kTestRecoveryKeyBasicCount);
    enableSOSCompatibilityForTests();
    secd_test_setup_temp_keychain(__FUNCTION__, NULL);
    testRecoveryKeyBasic();
    testRecoveryKey();
    secd_test_teardown_delete_temp_keychain(__FUNCTION__);
#else
    plan_tests(0);
#endif
    return 0;
}

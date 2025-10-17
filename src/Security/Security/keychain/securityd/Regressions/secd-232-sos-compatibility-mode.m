/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 2, 2023.
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
#include "secd_regressions.h"

#include "keychain/securityd/SecDbItem.h"
#include <utilities/array_size.h>
#include <utilities/SecCFWrappers.h>
#include <utilities/SecFileLocations.h>
#include <utilities/fileIo.h>

#include "keychain/securityd/SOSCloudCircleServer.h"
#include "keychain/securityd/SecItemServer.h"

#include <Security/SecBasePriv.h>

#include <TargetConditionals.h>
#include <AssertMacros.h>

#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <pthread.h>
#include "SecdTestKeychainUtilities.h"
#include <Security/OTConstants.h>

#include "SOSAccountTesting.h"

#if TARGET_OS_SIMULATOR || TARGET_OS_IPHONE
#if SOS_ENABLED

static void enableCompatibilityMode()
{
    CFErrorRef localError = NULL;
    bool setResult = SOSCCSetCompatibilityMode(true, &localError);
    ok(setResult == true, "result should be true");
    CFReleaseNull(localError);
    
    
    bool fetchResult = SOSCCFetchCompatibilityMode(&localError);
    ok(fetchResult == true, "result should be enabled");
    CFReleaseNull(localError);
}

static void disableCompatibilityMode()
{
    CFErrorRef localError = NULL;
    bool setResult = SOSCCSetCompatibilityMode(false, &localError);
    ok(setResult == true, "result should be true");
    CFReleaseNull(localError);
    
    
    bool fetchResult = SOSCCFetchCompatibilityMode(&localError);
    ok(fetchResult == false, "result should be disabled");
    ok(localError == nil, "error should be nil");
    CFReleaseNull(localError);
}

static void tests(void)
{
    CFErrorRef error = NULL;

    disableCompatibilityMode();
    
    bool joinResult = SOSCCRequestToJoinCircle(&error);
    ok(joinResult == false, "join result should be false");
    ok(error != nil, "error should not be nil");
    ok(CFErrorGetCode(error) == kSOSErrorPlatformNoSOS, "error code should be kSOSErrorPlatformNoSOS");
    
    NSString* description = CFBridgingRelease(CFErrorCopyDescription(error));
    ok(description, "description should not be nil");
    ok([description isEqualToString:@"The operation couldnâ€™t be completed. (com.apple.security.sos.error error 1050 - SOS Disabled for this platform)"], "error description should be SOS Disabled for this platform");
    description = nil;

    CFReleaseNull(error);
    
    enableCompatibilityMode();
    
    joinResult = SOSCCRequestToJoinCircle(&error);
    ok(joinResult == false, "join result should be false");
    ok (error != nil, "error should not be nil");
    
    ok(CFErrorGetCode(error) == kSOSErrorPrivateKeyAbsent, "error code should be kSOSErrorPrivateKeyAbsent");
    description = CFBridgingRelease(CFErrorCopyDescription(error));
    ok(description, "description should not be nil");
    
    ok([description isEqualToString:@"The operation couldnâ€™t be completed. (com.apple.security.sos.error error 1 - Private Key not available - failed to prompt user recently)"], "error description should be The operation couldnâ€™t be completed. (com.apple.security.sos.error error 1 - Private Key not available - failed to prompt user recently)");
    CFReleaseNull(error);
    
}
#endif

int secd_232_sos_compatibility_mode(int argc, char *const *argv)
{
#if SOS_ENABLED
    plan_tests(16);
    secd_test_setup_temp_keychain(__FUNCTION__, NULL);
    tests();
    secd_test_teardown_delete_temp_keychain(__FUNCTION__);
#else
    plan_tests(0);
#endif
    return 0;
}
#endif

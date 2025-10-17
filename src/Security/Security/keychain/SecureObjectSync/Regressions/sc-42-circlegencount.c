/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 7, 2022.
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
//  sc-42-circlegencount.c
//  sec
//
//  Created by Richard Murphy on 9/10/14.
//
//




#include <Security/SecBase.h>
#include <Security/SecItem.h>
#include <Security/SecKey.h>

#include "keychain/SecureObjectSync/SOSCircle.h"
#include <Security/SecureObjectSync/SOSCloudCircle.h>
#include <Security/SecureObjectSync/SOSPeerInfo.h>
#include "keychain/SecureObjectSync/SOSInternal.h"
#include "keychain/SecureObjectSync/SOSUserKeygen.h"

#include <utilities/SecCFWrappers.h>

#include <CoreFoundation/CoreFoundation.h>

#include <stdlib.h>
#include <unistd.h>

#include "keychain/securityd/SOSCloudCircleServer.h"

#include "SOSCircle_regressions.h"

#include "SOSRegressionUtilities.h"
#if SOS_ENABLED

static void tests(void)
{
    uint64_t beginvalue;
    uint64_t lastvalue;
    uint64_t newvalue;
    
    SOSCircleRef circle = SOSCircleCreate(NULL, CFSTR("TEST DOMAIN"), NULL);
    
    ok(NULL != circle, "Circle creation");
    
    ok(0 == SOSCircleCountPeers(circle), "Zero peers");
    
    ok(0 != (beginvalue = SOSCircleGetGenerationSint(circle))); // New circles should never be 0
    
    SOSCircleGenerationSetValue(circle, 0);
    
    ok(0 == SOSCircleGetGenerationSint(circle)); // Know we're starting out with a zero value (forced)
        
    SOSCircleGenerationIncrement(circle);
    
    ok(beginvalue <= (newvalue = SOSCircleGetGenerationSint(circle))); // incremented value should be greater or equal than where we began quantum is 2 seconds
    lastvalue = newvalue;
    
    SOSCircleGenerationIncrement(circle);
    ok(lastvalue < (newvalue = SOSCircleGetGenerationSint(circle))); // incremented value should be greater than last
    lastvalue = newvalue;

    SOSCircleResetToEmpty(circle, NULL);
    ok(lastvalue < (newvalue = SOSCircleGetGenerationSint(circle))); // incremented value should be greater than last
    
    CFReleaseNull(circle);
}
#endif

int sc_42_circlegencount(int argc, char *const *argv)
{
#if SOS_ENABLED
    plan_tests(7);
    tests();
#else
    plan_tests(0);
#endif
    return 0;
}

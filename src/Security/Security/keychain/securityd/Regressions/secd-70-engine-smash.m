/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
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
#include "keychain/SecureObjectSync/Regressions/SOSTestDevice.h"
#include "secd_regressions.h"
#include "SecdTestKeychainUtilities.h"
#include "SOSAccountTesting.h"

#if SOS_ENABLED

static int kTestTestCount = 581;

// Smash together identical items, but mute the devices every once in a while.
// (Simulating packet loss.)
static void smash(void) {
    __block int iteration=0;
    SOSTestDeviceListTestSync("smash", test_directive, test_reason, 0, true, ^bool(SOSTestDeviceRef source, SOSTestDeviceRef dest) {
        if (iteration < 100 && iteration % 10 == 0) {
            SOSTestDeviceSetMute(source, !SOSTestDeviceIsMute(source));
        }
        return false;
    }, ^bool(SOSTestDeviceRef source, SOSTestDeviceRef dest, SOSMessageRef message) {
        if (iteration++ < 200) {
            CFStringRef name = CFStringCreateWithFormat(NULL, NULL, CFSTR("smash-post-%d"), iteration);
            SOSTestDeviceAddGenericItem(source, name, name);
            SOSTestDeviceAddGenericItem(dest, name, name);
            CFReleaseNull(name);
            return true;
        }
        return false;
    }, CFSTR("alice"), CFSTR("bob"), NULL);
}
#endif

int secd_70_engine_smash(int argc, char *const *argv)
{
#if SOS_ENABLED
    plan_tests(kTestTestCount);
    enableSOSCompatibilityForTests();
    /* custom keychain dir */
    secd_test_setup_temp_keychain(__FUNCTION__, NULL);
    smash();
    secd_test_teardown_delete_temp_keychain(__FUNCTION__);
#else
    plan_tests(0);
#endif
    return 0;
}

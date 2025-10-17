/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 21, 2022.
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

static int kTestTestCount = 646;

// Add 1000 items on each device.  Then delete 1000 items on each device.
static void beer_servers(void) {
    __block int iteration=0;
    __block int objectix=0;
    __block CFMutableArrayRef objectNames = CFArrayCreateMutable(kCFAllocatorDefault, 0, &kCFTypeArrayCallBacks);
    const size_t itemDataSize = 1024;
    const int peerCount = 4;
    const int edgeCount = peerCount * (peerCount - 1);
    const int itemsPerPeer = 1000;
    const int itemsPerIteration = 100;
    const int itemChunkCount = (itemsPerPeer / itemsPerIteration);
    const int deleteIteration = edgeCount * (itemChunkCount + 30);
    const int idleIteration = 616; //deleteIteration + edgeCount * (itemChunkCount + 10);

    CFMutableDataRef itemData = CFDataCreateMutable(kCFAllocatorDefault, itemDataSize);
    CFDataSetLength(itemData, itemDataSize);
    SOSTestDeviceListTestSync("beer_servers", test_directive, test_reason, 0, false, ^bool(SOSTestDeviceRef source, SOSTestDeviceRef dest) {
        bool result = false;
        if (iteration <= edgeCount * itemChunkCount) {
            if (iteration % (peerCount - 1) == 0) {
                for (int j = 0; j < itemsPerIteration; ++j) {
                    CFStringRef name = CFStringCreateWithFormat(NULL, NULL, CFSTR("%@-pre-%d"), SOSTestDeviceGetID(source), objectix++);
                    CFArrayAppendValue(objectNames, name);
                    SOSTestDeviceAddGenericItemWithData(source, name, name, itemData);
                    CFReleaseNull(name);
                }
            }
            result = true;
        } else if (iteration == deleteIteration) {
            //diag("deletion starting");
        } else if (deleteIteration < iteration && iteration <= deleteIteration + edgeCount * itemChunkCount) {
            if (iteration % (peerCount - 1) == 0) {
                for (int j = 0; j < itemsPerIteration; ++j) {
                    CFStringRef name = CFArrayGetValueAtIndex(objectNames, --objectix);
                    SOSTestDeviceAddGenericItemTombstone(source, name, name);
                }
                result = true;
            }
        } else if (idleIteration == iteration) {
            //diag("idle starting");
        } else if (617 == iteration) {
            //diag("interesting");
        }
        iteration++;
        return result || iteration < deleteIteration;
    }, ^bool(SOSTestDeviceRef source, SOSTestDeviceRef dest, SOSMessageRef message) {
        return false;
    }, CFSTR("becks"), CFSTR("corona"), CFSTR("heineken"), CFSTR("spaten"), NULL);
    CFRelease(objectNames);
    CFRelease(itemData);
}
#endif

int secd_74_engine_beer_servers(int argc, char *const *argv)
{
#if SOS_ENABLED
    plan_tests(kTestTestCount);
    enableSOSCompatibilityForTests();
    /* custom keychain dir */
    secd_test_setup_temp_keychain(__FUNCTION__, NULL);
    beer_servers();
    secd_test_teardown_delete_temp_keychain(__FUNCTION__);
#else
    plan_tests(0);
#endif
    return 0;
}

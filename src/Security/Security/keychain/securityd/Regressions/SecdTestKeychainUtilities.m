/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 10, 2024.
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
#include "SecdTestKeychainUtilities.h"

#include <regressions/test/testmore.h>
#include <utilities/SecFileLocations.h>
#include <utilities/SecCFWrappers.h>
#include "keychain/securityd/SecItemServer.h"
#include <Security/SecureObjectSync/SOSViews.h>

#include "keychain/securityd/SecItemDataSource.h"

#import "Analytics/Clients/SOSAnalytics.h"
#import "Analytics/Clients/LocalKeychainAnalytics.h"
#import "keychain/ckks/CKKSAnalytics.h"

#include <CoreFoundation/CoreFoundation.h>

//#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

void secd_test_setup_temp_keychain(const char* test_prefix, dispatch_block_t do_in_reset)
{
    CFStringRef tmp_dir = CFStringCreateWithFormat(kCFAllocatorDefault, NULL, CFSTR("/tmp/%s.%X/"), test_prefix, arc4random());
    CFStringRef keychain_dir = CFStringCreateWithFormat(kCFAllocatorDefault, NULL, CFSTR("%@Library/Keychains"), tmp_dir);
    secnotice("secdtest", "Keychain path: %@", keychain_dir);
    
    CFStringPerformWithCString(keychain_dir, ^(const char *keychain_dir_string) {
        errno_t err = mkpath_np(keychain_dir_string, 0755);
        ok(err == 0 || err == EEXIST, "Create temp dir %s (%d)", keychain_dir_string, err);
    });
    
    
    /* set custom keychain dir, reset db */
    SecSetCustomHomeURLString(tmp_dir);

    SecKeychainDbReset(do_in_reset);

    CFReleaseNull(tmp_dir);
    CFReleaseNull(keychain_dir);
}

bool secd_test_teardown_delete_temp_keychain(const char* test_prefix)
{
    NSURL* keychainDir = (NSURL*)CFBridgingRelease(SecCopyHomeURL());

    // Drop analytics dbs here
    [[SOSAnalytics logger] removeStateAndUnlinkFile:NO];
    [[LocalKeychainAnalytics logger] removeStateAndUnlinkFile:NO];
    [[CKKSAnalytics logger] removeStateAndUnlinkFile:NO];

    secd_test_clear_testviews();
    SecItemDataSourceFactoryReleaseAll();
    SecKeychainDbForceClose();
    SecKeychainDbReset(NULL);

    // Only perform the desctructive step if the url matches what we expect!
    NSString* testName = [NSString stringWithUTF8String:test_prefix];

    if([keychainDir.path hasPrefix:[NSString stringWithFormat:@"/tmp/%@.", testName]]) {
        secnotice("secd_tests", "Removing test-specific keychain directory at %@", keychainDir);

        NSError* removeError = nil;
        [[NSFileManager defaultManager] removeItemAtURL:keychainDir error:&removeError];
        if(removeError) {
            secnotice("secd_tests", "Failed to remove directory: %@", removeError);
            return false;
        }

        return true;
     } else {
         secnotice("secd_tests", "Not removing keychain directory (%@), as it doesn't appear to be test-specific (for test %@)", keychainDir.path, testName);
         return false;
    }
}

CFStringRef kTestView1 = CFSTR("TestView1");
CFStringRef kTestView2 = CFSTR("TestView2");

void secd_test_setup_testviews(void) {    
    CFMutableSetRef testViews = CFSetCreateMutableForCFTypes(kCFAllocatorDefault);
    CFSetAddValue(testViews, kTestView1);
    CFSetAddValue(testViews, kTestView2);
    
    SOSViewsSetTestViewsSet(testViews);
    CFReleaseNull(testViews);
}

void secd_test_clear_testviews(void) {
    SOSViewsSetTestViewsSet(NULL);
}



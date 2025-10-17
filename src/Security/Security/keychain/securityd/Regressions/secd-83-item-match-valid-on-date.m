/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 9, 2022.
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
//  secd-83-item-match-valid-on-date.m
//  sec

#import <Foundation/Foundation.h>
#import <Security/SecItem.h>
#import <Security/SecBase.h>
#import <utilities/SecCFWrappers.h>


#import "secd_regressions.h"
#import "SecdTestKeychainUtilities.h"
#import "secd-83-item-match.h"

static void test(id returnKeyName) {
    NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
    [dateFormatter setDateFormat:@"yyyy-MM-dd HH:mm:ss zzz"];
    [dateFormatter setLocale:[[NSLocale alloc] initWithLocaleIdentifier:@"us_EN"]];
    NSDate *validDate = [dateFormatter dateFromString: @"2016-04-07 16:00:00 GMT"];
    NSDate *dateBefore = [dateFormatter dateFromString: @"2016-04-06 16:00:00 GMT"];
    NSDate *dateAfter = [dateFormatter dateFromString: @"2017-04-08 16:00:00 GMT"];

    CFTypeRef result = NULL;
    ok_status(SecItemCopyMatching( (__bridge CFDictionaryRef)@{ (id)kSecClass : (id)kSecClassCertificate,
                                                                (id)kSecMatchLimit : (id)kSecMatchLimitAll,
                                                                returnKeyName : @YES }, &result));
    ok(result && CFArrayGetCount(result) == 6);
    CFReleaseNull(result);

    ok_status(SecItemCopyMatching( (__bridge CFDictionaryRef)@{ (id)kSecClass : (id)kSecClassCertificate,
                                                                (id)kSecMatchLimit : (id)kSecMatchLimitAll,
                                                                (id)kSecMatchValidOnDate : validDate,
                                                                returnKeyName : @YES }, &result));
    ok(result && CFArrayGetCount(result) == 6);

    is_status(SecItemCopyMatching( (__bridge CFDictionaryRef)@{ (id)kSecClass : (id)kSecClassCertificate,
                                                                (id)kSecMatchLimit : (id)kSecMatchLimitAll,
                                                                (id)kSecMatchValidOnDate : dateBefore,
                                                                returnKeyName : @YES }, &result), errSecItemNotFound);
    ok(result && CFArrayGetCount(result) == 6);
    is_status(SecItemCopyMatching( (__bridge CFDictionaryRef)@{ (id)kSecClass : (id)kSecClassCertificate,
                                                                (id)kSecMatchLimit : (id)kSecMatchLimitAll,
                                                                (id)kSecMatchValidOnDate : dateAfter,
                                                                returnKeyName : @YES }, &result), errSecItemNotFound);
    ok(result && CFArrayGetCount(result) == 6);
}

int secd_83_item_match_valid_on_date(int argc, char *const *argv)
{
    secd_test_setup_temp_keychain(__FUNCTION__, NULL);
    plan_tests(39);

    @autoreleasepool {
    addTestCertificates();
        NSArray *returnKeyNames = @[(id)kSecReturnAttributes, (id)kSecReturnData, (id)kSecReturnRef, (id)kSecReturnPersistentRef];
        for (id returnKeyName in returnKeyNames)
            test(returnKeyName);
    }

    secd_test_teardown_delete_temp_keychain(__FUNCTION__);

    return 0;
}

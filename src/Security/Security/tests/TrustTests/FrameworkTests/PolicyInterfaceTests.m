/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 11, 2022.
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
#include <AssertMacros.h>
#import <XCTest/XCTest.h>
#include <Security/SecPolicyPriv.h>
#include <Security/SecPolicyInternal.h>
#include <utilities/array_size.h>
#include <utilities/SecCFWrappers.h>

#include "../TestMacroConversions.h"
#include "../TrustEvaluationTestHelpers.h"
#include "TrustFrameworkTestCase.h"

@interface PolicyInterfaceTests : TrustFrameworkTestCase
@end

@implementation PolicyInterfaceTests

- (void)testCreateWithProperties
{
    const void *keys[] = { kSecPolicyName, kSecPolicyClient };
    const void *values[] = { CFSTR("www.google.com"), kCFBooleanFalse };
    CFDictionaryRef properties = CFDictionaryCreate(NULL, keys, values,
            array_size(keys),
            &kCFTypeDictionaryKeyCallBacks,
            &kCFTypeDictionaryValueCallBacks);
    SecPolicyRef policy = SecPolicyCreateWithProperties(kSecPolicyAppleSSL, properties);
    isnt(policy, NULL, "SecPolicyCreateWithProperties");
    CFReleaseSafe(properties);
    CFReleaseSafe(policy);
}

- (void)testCopyProperties
{
    SecPolicyRef policy = SecPolicyCreateSSL(true, CFSTR("www.google.com"));
    CFDictionaryRef properties = NULL;
    isnt(properties = SecPolicyCopyProperties(policy), NULL, "copy policy properties");
    CFTypeRef value = NULL;
    is(CFDictionaryGetValueIfPresent(properties, kSecPolicyName, (const void **)&value) &&
        kCFCompareEqualTo == CFStringCompare((CFStringRef)value, CFSTR("www.google.com"), 0),
        true, "has policy name");
    is(CFDictionaryGetValueIfPresent(properties, kSecPolicyOid, (const void **)&value) &&
        CFEqual(value, kSecPolicyAppleSSL) , true, "has SSL policy");
    CFReleaseSafe(properties);
    CFReleaseSafe(policy);
}

- (void)testSetSHA256Pins
{
    SecPolicyRef policy = SecPolicyCreateBasicX509();
    CFDictionaryRef options = SecPolicyGetOptions(policy);
    XCTAssertEqual(CFDictionaryGetValue(options, kSecPolicyCheckLeafSPKISHA256), NULL);
    XCTAssertEqual(CFDictionaryGetValue(options, kSecPolicyCheckCAspkiSHA256), NULL);

    NSArray *pins = @[ ];
    SecPolicySetSHA256Pins(policy, (__bridge CFArrayRef)pins, (__bridge CFArrayRef)pins);
    XCTAssertEqualObjects((__bridge NSArray *)CFDictionaryGetValue(options, kSecPolicyCheckLeafSPKISHA256), pins);
    XCTAssertEqualObjects((__bridge NSArray *)CFDictionaryGetValue(options, kSecPolicyCheckCAspkiSHA256), pins);

    SecPolicySetSHA256Pins(policy, NULL, (__bridge CFArrayRef)pins);
    XCTAssertEqual(CFDictionaryGetValue(options, kSecPolicyCheckLeafSPKISHA256), NULL);
    XCTAssertEqualObjects((__bridge NSArray *)CFDictionaryGetValue(options, kSecPolicyCheckCAspkiSHA256), pins);

    SecPolicySetSHA256Pins(policy, (__bridge CFArrayRef)pins, NULL);
    XCTAssertEqualObjects((__bridge NSArray *)CFDictionaryGetValue(options, kSecPolicyCheckLeafSPKISHA256), pins);
    XCTAssertEqual(CFDictionaryGetValue(options, kSecPolicyCheckCAspkiSHA256), NULL);

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnonnull"
    [[clang::suppress]] SecPolicySetSHA256Pins(NULL, NULL, NULL);
    XCTAssertEqualObjects((__bridge NSArray *)CFDictionaryGetValue(options, kSecPolicyCheckLeafSPKISHA256), pins);
    XCTAssertEqual(CFDictionaryGetValue(options, kSecPolicyCheckCAspkiSHA256), NULL);
#pragma clang diagnostic pop
    
    CFReleaseSafe(policy);
}

@end

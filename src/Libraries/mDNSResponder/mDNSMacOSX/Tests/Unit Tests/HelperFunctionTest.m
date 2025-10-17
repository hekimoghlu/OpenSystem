/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 10, 2023.
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
#import <XCTest/XCTest.h>
#include "unittest_common.h"

@interface HelperFunctionTest : XCTestCase

@end

@implementation HelperFunctionTest

- (void)testCFStringToDomainLabel
{
    // test_cstring[i][0] is the input
    // test_cstring[i][1] is the expected correct output
    static const char * const test_cstring[][2] = {
        {"short", "short"},
        {"this-is-a-normal-computer-name", "this-is-a-normal-computer-name"},
        {"", ""},
        {"This is an ascii string whose length is more than 63 bytes, where it takes one byte to store every character", "This is an ascii string whose length is more than 63 bytes, whe"},
        {"à¤¯à¤¹ à¤à¤• à¤à¤¸à¥à¤¸à¥€ à¤¸à¥à¤Ÿà¥à¤°à¤¿à¤‚à¤— à¤¹à¥ˆ à¤œà¤¿à¤¸à¤•à¥€ à¤²à¤‚à¤¬à¤¾à¤ˆ à¤¸à¤¾à¤  à¤¤à¥€à¤¨ à¤¬à¤¾à¤‡à¤Ÿà¥à¤¸ à¤¸à¥‡ à¤…à¤§à¤¿à¤• à¤¹à¥ˆ, à¤œà¤¹à¤¾à¤‚ à¤¯à¤¹ à¤¹à¤° à¤šà¤°à¤¿à¤¤à¥à¤° à¤•à¥‹ à¤¸à¤‚à¤—à¥à¤°à¤¹à¥€à¤¤ à¤•à¤°à¤¨à¥‡ à¤•à¥‡ à¤²à¤¿à¤ à¤à¤• à¤¬à¤¾à¤‡à¤Ÿ à¤²à¥‡à¤¤à¤¾ à¤¹à¥ˆ", "à¤¯à¤¹ à¤à¤• à¤à¤¸à¥à¤¸à¥€ à¤¸à¥à¤Ÿà¥à¤°à¤¿à¤‚à¤— à¤¹à¥ˆ "}, // "à¤¯à¤¹ à¤à¤• à¤à¤¸à¥à¤¸à¥€ à¤¸à¥à¤Ÿà¥à¤°à¤¿à¤‚à¤— à¤¹à¥ˆ " is 62 byte, and "à¤¯à¤¹ à¤à¤• à¤à¤¸à¥à¤¸à¥€ à¤¸à¥à¤Ÿà¥à¤°à¤¿à¤‚à¤— à¤¹à¥ˆ à¤œà¤¿" is more than 63, so the result is expected to truncated to 62 bytes instead of 63 bytes
        {"à¤µà¤¿à¤¤à¥€à¤¯ à¤Ÿà¥‡à¤¸à¥à¤Ÿ à¤Ÿà¥à¤°à¤¾à¤ˆ à¤Ÿà¥€à¥°à¤µà¥€à¥°", "à¤µà¤¿à¤¤à¥€à¤¯ à¤Ÿà¥‡à¤¸à¥à¤Ÿ à¤Ÿà¥à¤°à¤¾à¤ˆ à¤Ÿà¥€à¥°à¤µà¥€"},
        {"è¿™æ˜¯ä¸€ä¸ªè¶…è¿‡å…­åä¸‰æ¯”ç‰¹çš„å…¶ä¸­æ¯ä¸ªä¸­æ–‡å­—ç¬¦å ä¸‰æ¯”ç‰¹çš„ä¸­æ–‡å­—ç¬¦ä¸²", "è¿™æ˜¯ä¸€ä¸ªè¶…è¿‡å…­åä¸‰æ¯”ç‰¹çš„å…¶ä¸­æ¯ä¸ªä¸­æ–‡å­—ç¬¦å "},
        {"ðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒ", "ðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒ"} // "ðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒ" is 60 bytes, and "ðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒðŸƒ" is more than 63 bytes so the result is expected to be truncated to 60 bytes instead of 64 bytes
    };

    for (size_t i = 0; i < sizeof(test_cstring) / sizeof(test_cstring[0]); i++) {
        // construct CFString from input
        CFStringRef name_ref = CFStringCreateWithCString(kCFAllocatorDefault, test_cstring[i][0], kCFStringEncodingUTF8);
        XCTAssertTrue(name_ref != NULL, @"unit test internal error. {descrption=\"name_ref should be non-NULL.\"}");

        // call the function being tested
        domainlabel label;
        mDNSDomainLabelFromCFString_ut(name_ref, &label);

        // Check if the result is correct
        XCTAssertEqual(label.c[0], strlen(test_cstring[i][1]),
                       @"name length is not equal. {expect=%lu,actual=%d}", strlen(test_cstring[i][1]), label.c[0]);
        XCTAssertTrue(memcmp(label.c + 1, test_cstring[i][1], label.c[0]) == 0,
                      @"name is not correctly decoded. {expect='%s',actual='%s'}", test_cstring[i][1], label.c + 1);

        CFRelease(name_ref);
    }
}

@end

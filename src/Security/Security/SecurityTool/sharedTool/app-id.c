/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 24, 2024.
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
//  app-id.c
//  Security
//

#import "app-id.h"

#import <stdio.h>
#import <CoreFoundation/CFDictionary.h>
#import <CoreFoundation/CFString.h>
#import <Security/SecItem.h>

int test_application_identifier(int argc, char * const *argv) {

    const void *keys[] = {kSecClass, kSecAttrService, kSecUseDataProtectionKeychain};
    const void *values[] = {kSecClassGenericPassword, NULL, kCFBooleanTrue};

    values[1] = CFStringCreateWithCStringNoCopy(NULL, "should-not-exist-testing-only", kCFStringEncodingUTF8, kCFAllocatorNull);

    CFDictionaryRef query = CFDictionaryCreate(NULL, keys, values, 3,
        &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);

    OSStatus status = SecItemCopyMatching(query, NULL);

    fprintf(stderr, "%d\n", (int)status);

    return status == errSecSuccess;
}

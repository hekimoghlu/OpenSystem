/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 19, 2021.
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
#include <Security/SecRecoveryPassword.h>

#include "keychain_regressions.h"

static void tests(void)
{
    const void *qs[] = {CFSTR("q1"), CFSTR("q2"), CFSTR("q3")};
    CFArrayRef questions = CFArrayCreate(kCFAllocatorDefault, qs, 3, NULL);
    
    const void *as[] = {CFSTR("a1"), CFSTR("a2"), CFSTR("a3")};
    CFArrayRef answers = CFArrayCreate(kCFAllocatorDefault, as, 3, NULL);
    
    CFStringRef password = CFSTR("AAAA-AAAA-AAAA-AAAA-AAAA-AAAA");
    
    CFDictionaryRef wrappedPassword = SecWrapRecoveryPasswordWithAnswers(password, questions, answers);
    isnt(wrappedPassword, NULL, "wrappedPassword NULL");
    
    CFStringRef recoveredPassword = SecUnwrapRecoveryPasswordWithAnswers(wrappedPassword, answers);
    isnt(recoveredPassword, NULL, "recoveredPassword NULL");

    is(CFStringCompare(password, recoveredPassword, 0), kCFCompareEqualTo, "SecRecoveryPassword");
    
    CFRelease(questions);
    CFRelease(answers);
    CFRelease(wrappedPassword);
    CFRelease(recoveredPassword);
}

int kc_44_secrecoverypassword(int argc, char *const *argv)
{
    plan_tests(3);
    tests();
    
    return 0;
}


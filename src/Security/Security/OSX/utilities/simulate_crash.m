/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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
//  simulate_crash
//  utilities
//
//  Copyright (c) 2014 Apple Inc. All Rights Reserved.
//

#import <TargetConditionals.h>

#include "debugging.h"

#import <mach/mach.h>
#import <SoftLinking/SoftLinking.h>
#import <Foundation/Foundation.h>

#if !TARGET_OS_SIMULATOR

SOFT_LINK_OPTIONAL_FRAMEWORK(PrivateFrameworks, CrashReporterSupport);

SOFT_LINK_FUNCTION(CrashReporterSupport, SimulateCrash, soft_SimulateCrash, \
                   BOOL, (pid_t pid, mach_exception_data_type_t exceptionCode, NSString *description),
                   (pid, exceptionCode, description));
SOFT_LINK_FUNCTION(CrashReporterSupport, WriteStackshotReport, soft_WriteStackshotReport, \
                   BOOL, (NSString *reason, mach_exception_data_type_t exceptionCode),
                   (reason, exceptionCode));

#endif // !TARGET_OS_SIMULATOR

static int __simulate_crash_counter = -1;

void __security_simulatecrash(CFStringRef reason, uint32_t code)
{
#if !TARGET_OS_SIMULATOR
    secerror("Simulating crash, reason: %@, code=%08x", reason, code);
    if (__security_simulatecrash_enabled() && isCrashReporterSupportAvailable()) {
        soft_SimulateCrash(getpid(), code, (__bridge NSString *)reason);
    } else {
        __simulate_crash_counter++;
    }
#else
    secerror("Simulating crash (not supported on simulator), reason: %@, code=%08x", reason, code);
#endif
}

void __security_stackshotreport(CFStringRef reason, uint32_t code)
{
#if !TARGET_OS_SIMULATOR
    secerror("stackshot report, reason: %@, code=%08x", reason, code);
    if (!__security_simulatecrash_enabled() && isCrashReporterSupportAvailable()) {
        return;
    }
    if (isCrashReporterSupportAvailable()) {
        soft_WriteStackshotReport((__bridge NSString *)reason, code);
    }
#else
    secerror("stackshot report (not supported on simulator, reason: %@, code=%08x", reason, code);
#endif
}


int __security_simulatecrash_enable(bool enable)
{
    int count = __simulate_crash_counter;
    __simulate_crash_counter = enable ? -1 : 0;
    return count;
}

bool __security_simulatecrash_enabled(void)
{
    return __simulate_crash_counter == -1;
}


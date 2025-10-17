/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 23, 2024.
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
#include "Logging.h"
#include "BPlatform.h"

#if !BUSE(OS_LOG)
#include <stdarg.h>
#include <stdio.h>
#endif

#if BPLATFORM(IOS_FAMILY)
#include <CoreFoundation/CoreFoundation.h>
#include <mach/exception_types.h>
#include <objc/objc.h>
#include <unistd.h>

#include "BSoftLinking.h"
BSOFT_LINK_PRIVATE_FRAMEWORK(CrashReporterSupport);
BSOFT_LINK_FUNCTION(CrashReporterSupport, SimulateCrash, BOOL, (pid_t pid, mach_exception_data_type_t exceptionCode, CFStringRef description), (pid, exceptionCode, description));
#endif

namespace bmalloc {

void logVMFailure(size_t vmSize)
{
#if BPLATFORM(IOS_FAMILY)
    const mach_exception_data_type_t kExceptionCode = 0xc105ca11;
    CFStringRef description = CFStringCreateWithFormat(kCFAllocatorDefault, nullptr, CFSTR("bmalloc failed to mmap %lu bytes"), vmSize);
    SimulateCrash(getpid(), kExceptionCode, description);
    CFRelease(description);
#else
    BUNUSED_PARAM(vmSize);
#endif
}

#if !BUSE(OS_LOG)
void reportAssertionFailureWithMessage(const char* file, int line, const char* function, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fprintf(stderr, "%s(%d) : %s\n", file, line, function);
}
#endif

} // namespace bmalloc

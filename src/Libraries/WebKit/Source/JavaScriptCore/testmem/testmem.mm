/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 25, 2023.
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
#import "config.h"
#import <JavaScriptCore/JavaScriptCore.h>
#import <inttypes.h>
#import <stdio.h>
#import <wtf/Compiler.h>
#import <wtf/text/StringToIntegerConversion.h>

#if __has_include(<libproc.h>)
#define HAS_LIBPROC 1
#import <libproc.h>
#else
#define HAS_LIBPROC 0
#endif

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

#if HAS_LIBPROC && RUSAGE_INFO_CURRENT >= 4 && JSC_OBJC_API_ENABLED
static void description()
{
    printf("usage \n testmem <path-to-file-to-run> [iterations]\n");
}

struct Footprint {
    uint64_t current;
    uint64_t peak;

    static Footprint now()
    {
        rusage_info_v4 rusage;
        if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&rusage)) {
            printf("Failure when calling rusage\n");
            exit(1);
        }

        return { rusage.ri_phys_footprint, rusage.ri_lifetime_max_phys_footprint };
    }
};

int main(int argc, char* argv[])
{
    if (argc < 2) {
        description();
        exit(1);
    }

    size_t iterations = 20;
    if (argc >= 3) {
        int iters = parseInteger<int>(unsafeSpan(argv[2])).value_or(0);
        if (iters < 0) {
            printf("Iterations argument must be >= 0");
            exit(1);
        }
        iterations = iters;
    }

    NSString *path = [NSString stringWithUTF8String:argv[1]];
    NSString *script = [[NSString alloc] initWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
    if (!script) {
        printf("Can't open file: %s\n", argv[1]);
        exit(1);
    }

    auto startTime = CFAbsoluteTimeGetCurrent();
    JSVirtualMachine* vm = [[JSVirtualMachine alloc] init];
    for (size_t i = 0; i < iterations; ++i) {
        @autoreleasepool {
            JSContext *context = [[JSContext alloc] initWithVirtualMachine:vm];
            context.exceptionHandler = ^(JSContext*, JSValue*) {
                printf("Unexpected exception thrown\n");
                exit(1);
            };
            [context evaluateScript:script];
        }
    }
    auto time = CFAbsoluteTimeGetCurrent() - startTime;
    auto footprint = Footprint::now();

    printf("time: %lf\n", time); // Seconds
    printf("peak footprint: %" PRIu64 "\n", footprint.peak); // Bytes
    printf("footprint at end: %" PRIu64 "\n", footprint.current); // Bytes

    return 0;
}
#else
int main(int, char*[])
{
    printf("You need to compile this file with an SDK that has RUSAGE_INFO_V4 or later\n");
    return 1;
}
#endif

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

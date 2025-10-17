/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 30, 2022.
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
#include "config.h"
#include <JavaScriptCore/JSBase.h>
#include <JavaScriptCore/JSContextRef.h>
#include <JavaScriptCore/JSStringRef.h>
#include <inttypes.h>
#include <optional>
#include <stdio.h>
#include <sys/time.h>
#include <vector>
#include <wtf/MonotonicTime.h>
#include <wtf/text/StringToIntegerConversion.h>

#if PLATFORM(PLAYSTATION)
#include <memory-extra/showmap.h>
#endif

static void description()
{
    printf("usage \n testmem <path-to-file-to-run> [iterations]\n");
}

struct Footprint {
    uint64_t current;
    uint64_t peak;

    static std::optional<Footprint> now()
    {
#if PLATFORM(PLAYSTATION)
        memory_extra::showmap::Result<4> result;
        auto* entry = result.entry("SceNKFastMalloc");
        result.collect();
        return Footprint {
            static_cast<uint64_t>(entry->rss),
            static_cast<uint64_t>(entry->vss)
        };
#else
#error "No testmem implementation for this platform."
#endif
    }
};

static JSStringRef readScript(const char* path)
{
    FILE* file = fopen(path, "r");
    if (!file)
        return nullptr;

    fseek(file, 0, SEEK_END);
    auto length = ftell(file);
    fseek(file, 0, SEEK_SET);

    std::vector<char> buffer(length + 1);
    fread(buffer.data(), length, 1, file);
    buffer[length] = 0;
    fclose(file);

    return JSStringCreateWithUTF8CString(buffer.data());
}

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

    const char* path = argv[1];
    auto script = readScript(path);
    if (!script) {
        printf("Can't open file: %s\n", path);
        exit(1);
    }

    auto sourceURL = JSStringCreateWithUTF8CString(path);

    auto startTime = MonotonicTime::now();
    JSContextGroupRef group = JSContextGroupCreate();
    for (size_t i = 0; i < iterations; ++i) {
        JSGlobalContextRef context = JSGlobalContextCreateInGroup(group, nullptr);
        JSValueRef exception = nullptr;
        JSEvaluateScript(context, script, nullptr, sourceURL, 0, &exception);
        if (exception) {
            printf("Unexpected exception thrown\n");
            exit(1);
        }
        JSGlobalContextRelease(context);
    }

    auto time = MonotonicTime::now() - startTime;
    if (auto footprint = Footprint::now()) {
        printf("time: %lf\n", time.seconds()); // Seconds
        printf("peak footprint: %" PRIu64 "\n", footprint->peak); // Bytes
        printf("footprint at end: %" PRIu64 "\n", footprint->current); // Bytes
    } else {
        printf("Failure when calling rusage\n");
        exit(1);
    }

    return 0;
}

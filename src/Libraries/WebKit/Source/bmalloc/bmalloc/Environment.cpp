/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 25, 2024.
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
#include "BPlatform.h"
#include "Environment.h"
#include "ProcessCheck.h"
#include <cstdlib>
#include <cstring>
#if BOS(DARWIN)
#include <mach-o/dyld.h>
#elif BOS(UNIX)
#include <dlfcn.h>
#endif

#if BOS(UNIX)
#include "valgrind.h"
#endif

#if BPLATFORM(IOS_FAMILY) && !BPLATFORM(MACCATALYST) && !BPLATFORM(IOS_FAMILY_SIMULATOR)
#define BUSE_CHECK_NANO_MALLOC 1
#else
#define BUSE_CHECK_NANO_MALLOC 0
#endif

#if BUSE(CHECK_NANO_MALLOC)
extern "C" {
#if __has_include(<malloc_private.h>)
#include <malloc_private.h>
#endif
int malloc_engaged_nano(void);
}
#endif

#if BUSE(LIBPAS)
#include "pas_status_reporter.h"
#endif

namespace bmalloc {

static bool isWebKitMallocForceEnabled()
{
    const char* value = getenv("WebKitMallocForceEnabled");
    return value ? atoi(value) : false;
}

static bool isMallocEnvironmentVariableImplyingSystemMallocSet()
{
    const char* list[] = {
        "Malloc",
        "MallocLogFile",
        "MallocGuardEdges",
        "MallocDoNotProtectPrelude",
        "MallocDoNotProtectPostlude",
        "MallocScribble",
        "MallocCheckHeapStart",
        "MallocCheckHeapEach",
        "MallocCheckHeapSleep",
        "MallocCheckHeapAbort",
        "MallocErrorAbort",
        "MallocCorruptionAbort",
        "MallocHelp"
    };
    size_t size = sizeof(list) / sizeof(const char*);
    
    for (size_t i = 0; i < size; ++i) {
        if (getenv(list[i]))
            return true;
    }

    // FIXME: Remove this once lite logging works with memgraph capture (rdar://109283870).
    const char* mallocStackLogging = getenv("MallocStackLogging");
    if (mallocStackLogging && !strcmp(mallocStackLogging, "lite"))
        return true;

    return false;
}

static bool isLibgmallocEnabled()
{
    char* variable = getenv("DYLD_INSERT_LIBRARIES");
    if (!variable)
        return false;
    if (!strstr(variable, "libgmalloc"))
        return false;
    return true;
}

static bool isSanitizerEnabled()
{
#if BOS(DARWIN)
    static const char sanitizerPrefix[] = "/libclang_rt.";
    static const char asanName[] = "asan_";
    static const char tsanName[] = "tsan_";
    uint32_t imageCount = _dyld_image_count();
    for (uint32_t i = 0; i < imageCount; ++i) {
        const char* imageName = _dyld_get_image_name(i);
        if (!imageName)
            continue;
        if (const char* s = strstr(imageName, sanitizerPrefix)) {
            const char* sanitizerName = s + sizeof(sanitizerPrefix) - 1;
            if (!strncmp(sanitizerName, asanName, sizeof(asanName) - 1))
                return true;
            if (!strncmp(sanitizerName, tsanName, sizeof(tsanName) - 1))
                return true;
        }
    }
    return false;
#elif BOS(UNIX)
    void* handle = dlopen(nullptr, RTLD_NOW);
    if (!handle)
        return false;
    bool result = !!dlsym(handle, "__asan_init") || !!dlsym(handle, "__tsan_init");
    dlclose(handle);
    return result;
#else
    return false;
#endif
}

static bool isRunningOnValgrind()
{
#if BOS(UNIX)
    if (RUNNING_ON_VALGRIND)
        return true;
#endif
    return false;
}

#if BUSE(CHECK_NANO_MALLOC)
static bool isNanoMallocEnabled()
{
    int result = !!malloc_engaged_nano();
    return result;
}
#endif

DEFINE_STATIC_PER_PROCESS_STORAGE(Environment);

Environment::Environment(const LockHolder&)
    : m_isDebugHeapEnabled(computeIsDebugHeapEnabled())
{
#if BUSE(LIBPAS)
    const char* statusReporter = getenv("WebKitPasStatusReporter");
    if (statusReporter) {
        unsigned enabled;
        if (sscanf(statusReporter, "%u", &enabled) == 1)
            pas_status_reporter_enabled = enabled;
    }
#endif
}

bool Environment::computeIsDebugHeapEnabled()
{
    if (isWebKitMallocForceEnabled())
        return false;
    if (isMallocEnvironmentVariableImplyingSystemMallocSet())
        return true;
    if (isLibgmallocEnabled())
        return true;
    if (isSanitizerEnabled())
        return true;
    if (isRunningOnValgrind())
        return true;

#if BUSE(CHECK_NANO_MALLOC)
    if (!isNanoMallocEnabled() && !shouldProcessUnconditionallyUseBmalloc())
        return true;
#endif

#if BENABLE_MALLOC_HEAP_BREAKDOWN
    return true;
#endif

    return false;
}

} // namespace bmalloc

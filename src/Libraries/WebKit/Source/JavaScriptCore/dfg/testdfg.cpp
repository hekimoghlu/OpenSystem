/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 7, 2025.
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

#include "JSCJSValueInlines.h"
// The above are needed before DFGAbstractValue.h
#include "DFGAbstractValue.h"
#include "InitializeThreading.h"
#include <wtf/DataLog.h>
#include <wtf/Threading.h>
#include <wtf/WTFProcess.h>
#include <wtf/text/StringCommon.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

// We don't have a NO_RETURN_DUE_TO_EXIT, nor should we. That's ridiculous.
static bool hiddenTruthBecauseNoReturnIsStupid() { return true; }

static void usage()
{
    dataLog("Usage: testdfg [<filter>]\n");
    if (hiddenTruthBecauseNoReturnIsStupid())
        exitProcess(1);
}

#if ENABLE(DFG_JIT)

using namespace JSC;
using namespace JSC::DFG;

namespace {

// Nothing fancy for now; we just use the existing WTF assertion machinery.
#define CHECK(x) do {                                                           \
        if (!!(x))                                                              \
            break;                                                              \
        WTFReportAssertionFailure(__FILE__, __LINE__, WTF_PRETTY_FUNCTION, #x); \
        CRASH();                                                                \
    } while (false)


#define RUN_NOW(test) do {                      \
        if (!shouldRun(#test))                  \
            break;                              \
        dataLog(#test "...\n");          \
        test;                                   \
        dataLog(#test ": OK!\n");        \
    } while (false)

static void testEmptyValueDoesNotValidateWithHeapTop()
{
    AbstractValue value;

    value.makeHeapTop();
    CHECK(!value.validateOSREntryValue(JSValue(), FlushedJSValue));

    value.makeBytecodeTop();
    CHECK(value.validateOSREntryValue(JSValue(), FlushedJSValue));
}

void run(const char* filter)
{
    auto shouldRun = [&] (const char* testName) -> bool {
        return !filter || WTF::findIgnoringASCIICaseWithoutLength(testName, filter) != WTF::notFound;
    };

    RUN_NOW(testEmptyValueDoesNotValidateWithHeapTop());
}

} // anonymous namespace

#else // ENABLE(DFG_JIT)

static void run(const char*)
{
    dataLog("DFG JIT is not enabled.\n");
}

#endif // ENABLE(DFG_JIT)

int main(int argc, char** argv)
{
    const char* filter = nullptr;
    switch (argc) {
    case 1:
        break;
    case 2:
        filter = argv[1];
        break;
    default:
        usage();
        break;
    }

    JSC::initialize();
    
    run(filter);

    return 0;
}

#if OS(WINDOWS)
extern "C" __declspec(dllexport) int WINAPI dllLauncherEntryPoint(int argc, const char* argv[])
{
    return main(argc, const_cast<char**>(argv));
}
#endif

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

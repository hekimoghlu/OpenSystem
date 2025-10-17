/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 8, 2024.
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
#pragma once

#include "Gate.h"
#include "Opcode.h"
#include "OptionsList.h"
#include "SecureARM64EHashPins.h"
#include <wtf/WTFConfig.h>

namespace JSC {

class ExecutableAllocator;
class FixedVMPoolExecutableAllocator;
class VM;

#if ENABLE(SEPARATED_WX_HEAP)
using JITWriteSeparateHeapsFunction = void (*)(off_t, const void*, size_t);
#endif

struct Config {
    static Config& singleton();

    static void disableFreezingForTesting() { g_wtfConfig.disableFreezingForTesting(); }
    JS_EXPORT_PRIVATE static void enableRestrictedOptions();
    static void finalize() { WTF::Config::finalize(); }

    static void configureForTesting()
    {
        WTF::setPermissionsOfConfigPage();
        disableFreezingForTesting();
        enableRestrictedOptions();
    }

    bool isPermanentlyFrozen() { return g_wtfConfig.isPermanentlyFrozen; }

    // All the fields in this struct should be chosen such that their
    // initial value is 0 / null / falsy because Config is instantiated
    // as a global singleton.
    // FIXME: We should use a placement new constructor from JSC::initialize so we can use default initializers.

    bool restrictedOptionsEnabled;
    bool jitDisabled;
    bool vmCreationDisallowed;
    bool vmEntryDisallowed;

    bool useFastJITPermissions;

    // The following HasBeenCalled flags are for auditing call_once initialization functions.
    bool initializeHasBeenCalled;

    struct {
#if ASSERT_ENABLED
        bool canUseJITIsSet;
#endif
        bool canUseJIT;
    } vm;

#if CPU(ARM64E)
    bool canUseFPAC;
#endif

    ExecutableAllocator* executableAllocator;
    FixedVMPoolExecutableAllocator* fixedVMPoolExecutableAllocator;
    void* startExecutableMemory;
    void* endExecutableMemory;
    uintptr_t startOfFixedWritableMemoryPool;
    uintptr_t startOfStructureHeap;
    uintptr_t sizeOfStructureHeap;
    void* defaultCallThunk;
    void* arityFixupThunk;

#if ENABLE(SEPARATED_WX_HEAP)
    JITWriteSeparateHeapsFunction jitWriteSeparateHeaps;
#endif

    OptionsStorage options;

    void (*shellTimeoutCheckCallback)(VM&);

    struct {
        uint8_t exceptionInstructions[maxBytecodeStructLength + 1];
        uint8_t wasmExceptionInstructions[maxBytecodeStructLength + 1];
        const void* gateMap[numberOfGates];
    } llint;

#if CPU(ARM64E) && ENABLE(PTRTAG_DEBUGGING)
    WTF::PtrTagLookup ptrTagLookupRecord;
#endif

#if CPU(ARM64E) && ENABLE(JIT)
    SecureARM64EHashPins arm64eHashPins;
#endif
};

constexpr size_t alignmentOfJSCConfig = std::alignment_of<JSC::Config>::value;

static_assert(WTF::offsetOfWTFConfigExtension + sizeof(JSC::Config) <= WTF::ConfigSizeToProtect);
static_assert(roundUpToMultipleOf<alignmentOfJSCConfig>(WTF::offsetOfWTFConfigExtension) == WTF::offsetOfWTFConfigExtension);

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

// Workaround to localize bounds safety warnings to this file.
// FIXME: Use real types to make materializing JSC::Config* bounds-safe and type-safe.
inline Config* addressOfJSCConfig() { return std::bit_cast<Config*>(&g_wtfConfig.spaceForExtensions); }

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#define g_jscConfig (*JSC::addressOfJSCConfig())

constexpr size_t offsetOfJSCConfigInitializeHasBeenCalled = offsetof(JSC::Config, initializeHasBeenCalled);
constexpr size_t offsetOfJSCConfigGateMap = offsetof(JSC::Config, llint.gateMap);
constexpr size_t offsetOfJSCConfigStartOfStructureHeap = offsetof(JSC::Config, startOfStructureHeap);
constexpr size_t offsetOfJSCConfigDefaultCallThunk = offsetof(JSC::Config, defaultCallThunk);

ALWAYS_INLINE PURE_FUNCTION uintptr_t startOfStructureHeap()
{
    return g_jscConfig.startOfStructureHeap;
}

} // namespace JSC

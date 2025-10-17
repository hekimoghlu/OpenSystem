/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 15, 2022.
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
#include "InitializeThreading.h"

#include "AssemblyComments.h"
#include "AssertInvariants.h"
#include "ExecutableAllocator.h"
#include "InPlaceInterpreter.h"
#include "JITOperationList.h"
#include "JSCConfig.h"
#include "JSCPtrTag.h"
#include "LLIntData.h"
#include "NativeCalleeRegistry.h"
#include "Options.h"
#include "StructureAlignedMemoryAllocator.h"
#include "SuperSampler.h"
#include "VMTraps.h"
#include "WasmCapabilities.h"
#include "WasmFaultSignalHandler.h"
#include "WasmThunks.h"
#include <mutex>
#include <wtf/Threading.h>
#include <wtf/threads/Signals.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN

#if !USE(SYSTEM_MALLOC)
#include <bmalloc/BPlatform.h>
#if BUSE(LIBPAS)
#include <bmalloc/pas_scavenger.h>
#endif
#endif

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

#if ENABLE(LLVM_PROFILE_GENERATION)
extern "C" char __llvm_profile_filename[] = "/private/tmp/WebKitPGO/JavaScriptCore_%m_pid%p%c.profraw";
#endif

namespace JSC {

static_assert(sizeof(bool) == 1, "LLInt and JIT assume sizeof(bool) is always 1 when touching it directly from assembly code.");

enum class JSCProfileTag { };

void initialize()
{
    static std::once_flag onceFlag;

    std::call_once(onceFlag, [] {
        WTF::initialize();
        Options::initialize();

        initializePtrTagLookup();

#if ENABLE(WRITE_BARRIER_PROFILING)
        WriteBarrierCounters::initialize();
#endif
        {
            Options::AllowUnfinalizedAccessScope scope;
            JITOperationList::initialize();
            ExecutableAllocator::initialize();
            VM::computeCanUseJIT();
            if (!g_jscConfig.vm.canUseJIT) {
                Options::useJIT() = false;
                Options::notifyOptionsChanged();
            } else {
#if CPU(ARM64E) && ENABLE(JIT)
                g_jscConfig.arm64eHashPins.initializeAtStartup();
                isARM64E_FPAC(); // Call this to initialize g_jscConfig.canUseFPAC.
#endif
            }
            StructureAlignedMemoryAllocator::initializeStructureAddressSpace();
        }
        Options::finalize();

#if !USE(SYSTEM_MALLOC)
#if BUSE(LIBPAS)
        if (Options::libpasScavengeContinuously())
            pas_scavenger_disable_shut_down();
#endif
#endif

        JITOperationList::populatePointersInJavaScriptCore();

        AssemblyCommentRegistry::initialize();
#if ENABLE(WEBASSEMBLY)
        if (Options::useWasmIPInt())
            IPInt::initialize();
#endif
        LLInt::initialize();
        DisallowGC::initialize();

        initializeSuperSampler();
        Thread& thread = Thread::current();
        thread.setSavedLastStackTop(thread.stack().origin());

        NativeCalleeRegistry::initialize();
#if ENABLE(WEBASSEMBLY) && ENABLE(JIT)
        if (Wasm::isSupported()) {
            Wasm::Thunks::initialize();
        }
#endif

        if (VM::isInMiniMode())
            WTF::fastEnableMiniMode();

        if (Wasm::isSupported() || !Options::usePollingTraps()) {
            if (!Options::usePollingTraps())
                VMTraps::initializeSignals();
            if (Wasm::isSupported())
                Wasm::prepareSignalingMemory();
        }

        assertInvariants();

        WTF::compilerFence();
        RELEASE_ASSERT(!g_jscConfig.initializeHasBeenCalled);
        g_jscConfig.initializeHasBeenCalled = true;
    });
}

} // namespace JSC

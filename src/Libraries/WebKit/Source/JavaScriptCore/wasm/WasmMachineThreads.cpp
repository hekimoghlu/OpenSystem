/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 12, 2022.
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
#include "WasmMachineThreads.h"

#if ENABLE(WEBASSEMBLY)

#include "MachineStackMarker.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/ThreadMessage.h>

namespace JSC { namespace Wasm {


inline MachineThreads& wasmThreads()
{
    static LazyNeverDestroyed<MachineThreads> threads;
    static std::once_flag once;
    std::call_once(once, [] {
        threads.construct();
    });

    return threads;
}

void startTrackingCurrentThread()
{
    wasmThreads().addCurrentThread();
}

void resetInstructionCacheOnAllThreads()
{
    Locker locker { wasmThreads().getLock() };
    ThreadSuspendLocker threadSuspendLocker;
    for (auto& thread : wasmThreads().threads(locker)) {
        sendMessage(threadSuspendLocker, thread.get(), [] (const PlatformRegisters&) {
            // It's likely that the signal handler will already reset the instruction cache but we might as well be sure.
            WTF::crossModifyingCodeFence();
        });
    }
}

    
} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)

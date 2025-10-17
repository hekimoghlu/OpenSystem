/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 25, 2023.
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

#if ENABLE(WEBASSEMBLY) && ENABLE(JIT)

#include "MacroAssemblerCodeRef.h"
#include "WasmJS.h"
#include <wtf/TZoneMalloc.h>

namespace JSC { namespace Wasm {

typedef MacroAssemblerCodeRef<JITThunkPtrTag> (*ThunkGenerator)(const AbstractLocker&);

MacroAssemblerCodeRef<JITThunkPtrTag> throwExceptionFromWasmThunkGenerator(const AbstractLocker&);
MacroAssemblerCodeRef<JITThunkPtrTag> throwStackOverflowFromWasmThunkGenerator(const AbstractLocker&);
MacroAssemblerCodeRef<JITThunkPtrTag> catchInWasmThunkGenerator(const AbstractLocker&);
MacroAssemblerCodeRef<JITThunkPtrTag> crashDueToBBQStackOverflowGenerator(const AbstractLocker&);
MacroAssemblerCodeRef<JITThunkPtrTag> crashDueToOMGStackOverflowGenerator(const AbstractLocker&);
#if ENABLE(WEBASSEMBLY_OMGJIT)
MacroAssemblerCodeRef<JITThunkPtrTag> triggerOMGEntryTierUpThunkGeneratorImpl(const AbstractLocker&, bool isSIMDContext);
MacroAssemblerCodeRef<JITThunkPtrTag> triggerOMGEntryTierUpThunkGeneratorSIMD(const AbstractLocker&);
MacroAssemblerCodeRef<JITThunkPtrTag> triggerOMGEntryTierUpThunkGeneratorNoSIMD(const AbstractLocker&);
constexpr ThunkGenerator triggerOMGEntryTierUpThunkGenerator(bool isSIMDContext)
{
    if (isSIMDContext)
        return triggerOMGEntryTierUpThunkGeneratorSIMD;
    return triggerOMGEntryTierUpThunkGeneratorNoSIMD;
}
#endif

class Thunks {
    WTF_MAKE_TZONE_ALLOCATED(Thunks);
    WTF_MAKE_NONCOPYABLE(Thunks);
public:
    static void initialize();
    static Thunks& singleton();

    MacroAssemblerCodeRef<JITThunkPtrTag> stub(ThunkGenerator);
    MacroAssemblerCodeRef<JITThunkPtrTag> stub(const AbstractLocker&, ThunkGenerator);
    MacroAssemblerCodeRef<JITThunkPtrTag> existingStub(ThunkGenerator);

private:
    Thunks() = default;

    UncheckedKeyHashMap<ThunkGenerator, MacroAssemblerCodeRef<JITThunkPtrTag>> m_stubs;
    Lock m_lock;
};

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY) && ENABLE(JIT)

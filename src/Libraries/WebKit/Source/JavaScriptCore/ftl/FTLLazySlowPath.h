/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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

#if ENABLE(FTL_JIT)

#include "CCallHelpers.h"
#include "CodeBlock.h"
#include "CodeLocation.h"
#include "GPRInfo.h"
#include "MacroAssemblerCodeRef.h"
#include "RegisterSet.h"
#include "ScratchRegisterAllocator.h"
#include <wtf/SharedTask.h>
#include <wtf/TZoneMalloc.h>

namespace JSC { namespace FTL {

// A LazySlowPath is an object that represents a piece of code that is part of FTL generated code
// that will be generated lazily. It holds all of the important information needed to generate that
// code, such as where to link jumps to and which registers are in use. It also has a reference to a
// SharedTask that will do the actual code generation. That SharedTask may have additional data, like
// which registers hold the inputs or outputs.
class LazySlowPath {
    WTF_MAKE_NONCOPYABLE(LazySlowPath);
    WTF_MAKE_TZONE_ALLOCATED(LazySlowPath);
public:
    struct GenerationParams {
        // Extra parameters to the GeneratorFunction are made into fields of this struct, so that if
        // we add new parameters, we don't have to change all of the users.
        CCallHelpers::JumpList doneJumps;
        CCallHelpers::JumpList* exceptionJumps;
        LazySlowPath* lazySlowPath;
    };

    typedef void GeneratorFunction(CCallHelpers&, GenerationParams&);
    typedef SharedTask<GeneratorFunction> Generator;

    template<typename Functor>
    static Ref<Generator> createGenerator(const Functor& functor)
    {
        return createSharedTask<GeneratorFunction>(functor);
    }

    LazySlowPath() = default;

    ~LazySlowPath();

    void initialize(
        CodeLocationJump<JSInternalPtrTag> patchableJump, CodeLocationLabel<JSInternalPtrTag> done,
        CodeLocationLabel<ExceptionHandlerPtrTag> exceptionTarget, const RegisterSetBuilder& usedRegisters,
        CallSiteIndex, RefPtr<Generator>
        );

    CodeLocationJump<JSInternalPtrTag> patchableJump() const { return m_patchableJump; }
    CodeLocationLabel<JSInternalPtrTag> done() const { return m_done; }
    const ScalarRegisterSet& usedRegisters() const { return m_usedRegisters; }
    CallSiteIndex callSiteIndex() const { return m_callSiteIndex; }

    void generate(CodeBlock*);

    MacroAssemblerCodeRef<JITStubRoutinePtrTag> stub() const { return m_stub; }

private:
    CodeLocationJump<JSInternalPtrTag> m_patchableJump;
    CodeLocationLabel<JSInternalPtrTag> m_done;
    CodeLocationLabel<ExceptionHandlerPtrTag> m_exceptionTarget;
    ScalarRegisterSet m_usedRegisters;
    CallSiteIndex m_callSiteIndex;
    MacroAssemblerCodeRef<JITStubRoutinePtrTag> m_stub;
    RefPtr<Generator> m_generator;
};

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)

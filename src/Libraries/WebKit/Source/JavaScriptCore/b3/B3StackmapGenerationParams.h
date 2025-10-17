/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 1, 2024.
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

#if ENABLE(B3_JIT)

#include "AirGenerationContext.h"
#include "B3ValueRep.h"
#include "MacroAssembler.h"
#include "RegisterSet.h"
#include <wtf/Box.h>

namespace JSC {

class CCallHelpers;

namespace B3 {

class CheckSpecial;
class PatchpointSpecial;
class Procedure;
class StackmapValue;

// NOTE: It's possible to capture StackmapGenerationParams by value, but not all of the methods will
// work if you do that.
class StackmapGenerationParams {
public:
    // This is the stackmap value that we're generating.
    StackmapValue* value() const { return m_value; }
    
    // This tells you the actual value representations that were chosen. This is usually different
    // from the constraints we supplied.
    const Vector<ValueRep>& reps() const { return m_reps; };

    // Usually we wish to access the reps. We make this easy by making ourselves appear to be a
    // collection of reps.
    unsigned size() const { return m_reps.size(); }
    const ValueRep& at(unsigned index) const { return m_reps[index]; }
    const ValueRep& operator[](unsigned index) const { return at(index); }
    Vector<ValueRep>::const_iterator begin() const { return m_reps.begin(); }
    Vector<ValueRep>::const_iterator end() const { return m_reps.end(); }
    
    // This tells you the registers that were used.
    // NOTE: This will report bogus information if you did proc.setNeedsUsedRegisters(false).
    const RegisterSetBuilder& usedRegisters() const;

    // This is a useful helper if you want to do register allocation inside of a patchpoint. The
    // usedRegisters() set is not directly useful for this purpose because:
    //
    // - You can only use callee-save registers for scratch if they were saved in the prologue. So,
    //   if a register is callee-save, it's not enough that it's not in usedRegisters().
    //
    // - Scratch registers are going to be in usedRegisters() at the patchpoint. So, if you want to
    //   find one of your requested scratch registers using usedRegisters(), you'll have a bad time.
    //
    // This gives you the used register set that's useful for allocating scratch registers. This set
    // is defined as:
    //
    //     (usedRegisters() | (RegisterSetBuilder::calleeSaveRegisters() - proc.calleeSaveRegisters()))
    //     - gpScratchRegisters - fpScratchRegisters
    //
    // I.e. it is like usedRegisters() but also includes unsaved callee-saves and excludes scratch
    // registers.
    //
    // NOTE: This will report bogus information if you did proc.setNeedsUsedRegisters(false).
    JS_EXPORT_PRIVATE RegisterSetBuilder unavailableRegisters() const;

    GPRReg gpScratch(unsigned index) const { return m_gpScratch[index]; }
    FPRReg fpScratch(unsigned index) const { return m_fpScratch[index]; }
    
    // This is computed lazily, so it won't work if you capture StackmapGenerationParams by value.
    // These labels will get populated before any late paths or link tasks execute.
    JS_EXPORT_PRIVATE Vector<Box<MacroAssembler::Label>> successorLabels() const;
    
    // This is computed lazily, so it won't work if you capture StackmapGenerationParams by value.
    // Returns true if the successor at the given index is going to be emitted right after the
    // patchpoint.
    JS_EXPORT_PRIVATE bool fallsThroughToSuccessor(unsigned successorIndex) const;

    // These are provided for convenience; they mean that you don't have to capture them if you don't want to.
    JS_EXPORT_PRIVATE Procedure& proc() const;
    JS_EXPORT_PRIVATE Air::Code& code() const;
    
    // The Air::GenerationContext gives you even more power.
    Air::GenerationContext& context() const { return m_context; };

    template<typename Functor>
    void addLatePath(const Functor& functor) const
    {
        context().latePaths.append(
            createSharedTask<Air::GenerationContext::LatePathFunction>(
                [=] (CCallHelpers& jit, Air::GenerationContext&) {
                    functor(jit);
                }));
    }

private:
    friend class CheckSpecial;
    friend class PatchpointSpecial;
    
    StackmapGenerationParams(StackmapValue*, const Vector<ValueRep>& reps, Air::GenerationContext&);

    StackmapValue* m_value;
    Vector<ValueRep> m_reps;
    Vector<GPRReg> m_gpScratch;
    Vector<FPRReg> m_fpScratch;
    Air::GenerationContext& m_context;
};

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

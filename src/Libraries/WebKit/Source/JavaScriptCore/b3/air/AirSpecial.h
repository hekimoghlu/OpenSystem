/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 6, 2023.
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

#include "AirInst.h"
#include "B3SparseCollection.h"
#include <wtf/FastMalloc.h>
#include <wtf/Noncopyable.h>
#include <wtf/ScopedLambda.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/CString.h>

namespace JSC { namespace B3 { namespace Air {

class Code;
struct GenerationContext;

class Special {
    WTF_MAKE_NONCOPYABLE(Special);
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(Special, JS_EXPORT_PRIVATE);
public:
    static const char* const dumpPrefix;
    
    Special();
    JS_EXPORT_PRIVATE virtual ~Special();

    Code& code() const { return *m_code; }

    CString name() const;

    virtual void forEachArg(Inst&, const ScopedLambda<Inst::EachArgCallback>&) = 0;
    virtual bool isValid(Inst&) = 0;
    virtual bool admitsStack(Inst&, unsigned argIndex) = 0;
    virtual bool admitsExtendedOffsetAddr(Inst&, unsigned argIndex) = 0;
    virtual std::optional<unsigned> shouldTryAliasingDef(Inst&);

    // This gets called on for each Inst that uses this Special. Note that there is no way to
    // guarantee that a Special gets used from just one Inst, because Air might taildup late. So,
    // if you want to pass this information down to generate(), then you have to either:
    //
    // 1) Generate Air that starts with a separate Special per Patch Inst, and then merge
    //    usedRegister sets. This is probably not great, but it optimizes for the common case that
    //    Air didn't duplicate code or that such duplication didn't cause any interesting changes to
    //    register assignment.
    //
    // 2) Have the Special maintain a UncheckedKeyHashMap<Inst*, RegisterSetBuilder>. This works because the analysis
    //    that feeds into this call is performed just before code generation and there is no way
    //    for the Vector<>'s that contain the Insts to be reallocated. This allows generate() to
    //    consult the UncheckedKeyHashMap.
    //
    // 3) Hybrid: you could use (1) and fire up a UncheckedKeyHashMap if you see multiple calls.
    //
    // Note that it's not possible to rely on reportUsedRegisters() being called in the same order
    // as generate(). If we could rely on that, then we could just have each Special instance
    // maintain a Vector of RegisterSetBuilder's and then process that vector in the right order in
    // generate(). But, the ordering difference is unlikely to change since it would harm the
    // performance of the liveness analysis.
    //
    // Currently, we do (1) for B3 stackmaps.
    virtual void reportUsedRegisters(Inst&, const RegisterSetBuilder&) = 0;
    
    virtual MacroAssembler::Jump generate(Inst&, CCallHelpers&, GenerationContext&) = 0;

    virtual RegisterSetBuilder extraEarlyClobberedRegs(Inst&) = 0;
    virtual RegisterSetBuilder extraClobberedRegs(Inst&) = 0;
    
    // By default, this returns false.
    virtual bool isTerminal(Inst&);

    // By default, this returns true.
    virtual bool hasNonArgEffects(Inst&);

    // By default, this returns true.
    virtual bool hasNonArgNonControlEffects(Inst&);

    void dump(PrintStream&) const;
    void deepDump(PrintStream&) const;

protected:
    virtual void dumpImpl(PrintStream&) const = 0;
    virtual void deepDumpImpl(PrintStream&) const = 0;

private:
    friend class Code;
    friend class SparseCollection<Special>;

    unsigned m_index { UINT_MAX };
    Code* m_code { nullptr };
};

class DeepSpecialDump {
public:
    DeepSpecialDump(const Special* special)
        : m_special(special)
    {
    }

    void dump(PrintStream& out) const
    {
        if (m_special)
            m_special->deepDump(out);
        else
            out.print("<null>");
    }

private:
    const Special* m_special;
};

inline DeepSpecialDump deepDump(const Special* special)
{
    return DeepSpecialDump(special);
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)

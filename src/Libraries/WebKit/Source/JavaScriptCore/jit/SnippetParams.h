/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 8, 2023.
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

#if ENABLE(JIT)

#include "CCallHelpers.h"
#include "SnippetReg.h"
#include "SnippetSlowPathCalls.h"

namespace JSC {

class SnippetParams {
WTF_MAKE_NONCOPYABLE(SnippetParams);
public:
    virtual ~SnippetParams() { }

    class Value {
    public:
        Value(SnippetReg reg)
            : m_reg(reg)
        {
        }

        Value(SnippetReg reg, JSValue value)
            : m_reg(reg)
            , m_value(value)
        {
        }

        bool isGPR() const { return m_reg.isGPR(); }
        bool isFPR() const { return m_reg.isFPR(); }
        bool isJSValueRegs() const { return m_reg.isJSValueRegs(); }
        GPRReg gpr() const { return m_reg.gpr(); }
        FPRReg fpr() const { return m_reg.fpr(); }
        JSValueRegs jsValueRegs() const { return m_reg.jsValueRegs(); }

        SnippetReg reg() const
        {
            return m_reg;
        }

        JSValue value() const
        {
            return m_value;
        }

    private:
        SnippetReg m_reg;
        JSValue m_value;
    };

    unsigned size() const { return m_regs.size(); }
    const Value& at(unsigned index) const { return m_regs[index]; }
    const Value& operator[](unsigned index) const { return at(index); }

    GPRReg gpScratch(unsigned index) const { return m_gpScratch[index]; }
    FPRReg fpScratch(unsigned index) const { return m_fpScratch[index]; }

    SnippetParams(VM& vm, Vector<Value>&& regs, Vector<GPRReg>&& gpScratch, Vector<FPRReg>&& fpScratch)
        : m_vm(vm)
        , m_regs(WTFMove(regs))
        , m_gpScratch(WTFMove(gpScratch))
        , m_fpScratch(WTFMove(fpScratch))
    {
    }

    VM& vm() { return m_vm; }

    template<typename FunctionType, typename ResultType, typename... Arguments>
    void addSlowPathCall(CCallHelpers::JumpList from, CCallHelpers& jit, FunctionType function, ResultType result, Arguments... arguments)
    {
        addSlowPathCallImpl(from, jit, function, result, std::make_tuple(arguments...));
    }

private:
#define JSC_DEFINE_CALL_OPERATIONS(OperationType, ResultType, ...) JS_EXPORT_PRIVATE virtual void addSlowPathCallImpl(CCallHelpers::JumpList, CCallHelpers&, OperationType, ResultType, std::tuple<__VA_ARGS__> args) = 0;
    SNIPPET_SLOW_PATH_CALLS(JSC_DEFINE_CALL_OPERATIONS)
#undef JSC_DEFINE_CALL_OPERATIONS

    VM& m_vm;
    Vector<Value> m_regs;
    Vector<GPRReg> m_gpScratch;
    Vector<FPRReg> m_fpScratch;
};

}

#endif

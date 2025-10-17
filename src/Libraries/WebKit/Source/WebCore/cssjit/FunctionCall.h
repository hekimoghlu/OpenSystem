/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 6, 2023.
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

#if ENABLE(CSS_SELECTOR_JIT)

#include "RegisterAllocator.h"
#include "StackAllocator.h"
#include <JavaScriptCore/GPRInfo.h>
#include <JavaScriptCore/JSCPtrTag.h>
#include <JavaScriptCore/MacroAssembler.h>

namespace WebCore {

class FunctionCall {
public:
    FunctionCall(JSC::MacroAssembler& assembler, RegisterAllocator& registerAllocator, StackAllocator& stackAllocator, Vector<std::pair<JSC::MacroAssembler::Call, CodePtr<JSC::OperationPtrTag>>, 32>& callRegistry)
        : m_assembler(assembler)
        , m_registerAllocator(registerAllocator)
        , m_stackAllocator(stackAllocator)
        , m_callRegistry(callRegistry)
        , m_argumentCount(0)
        , m_firstArgument(JSC::InvalidGPRReg)
        , m_secondArgument(JSC::InvalidGPRReg)
    {
    }

    void setFunctionAddress(CodePtr<JSC::OperationPtrTag> functionAddress)
    {
        m_functionAddress = functionAddress;
    }

    void setOneArgument(const JSC::MacroAssembler::RegisterID& registerID)
    {
        m_argumentCount = 1;
        m_firstArgument = registerID;
    }

    void setTwoArguments(const JSC::MacroAssembler::RegisterID& firstRegisterID, const JSC::MacroAssembler::RegisterID& secondRegisterID)
    {
        m_argumentCount = 2;
        m_firstArgument = firstRegisterID;
        m_secondArgument = secondRegisterID;
    }

    void call()
    {
        prepareAndCall();
        cleanupPostCall();
    }

    JSC::MacroAssembler::Jump callAndBranchOnBooleanReturnValue(JSC::MacroAssembler::ResultCondition condition)
    {
#if CPU(X86_64)
        return callAndBranchOnCondition(condition, JSC::MacroAssembler::TrustedImm32(0xff));
#elif CPU(ARM64) || CPU(ARM)
        return callAndBranchOnCondition(condition, JSC::MacroAssembler::TrustedImm32(-1));
#else
#error Missing implementationg for matching boolean return values.
#endif
    }

private:
    JSC::MacroAssembler::Jump callAndBranchOnCondition(JSC::MacroAssembler::ResultCondition condition, JSC::MacroAssembler::TrustedImm32 mask)
    {
        prepareAndCall();
        m_assembler.test32(JSC::GPRInfo::returnValueGPR, mask);
        cleanupPostCall();
        return m_assembler.branch(condition);
    }

    void swapArguments()
    {
        JSC::MacroAssembler::RegisterID a = m_firstArgument;
        JSC::MacroAssembler::RegisterID b = m_secondArgument;
        // x86 can swap without a temporary register. On other architectures, we need allocate a temporary register to switch the values.
#if CPU(X86_64)
        m_assembler.swap(a, b);
#elif CPU(ARM64) || CPU(ARM_THUMB2)
        m_assembler.move(a, tempRegister);
        m_assembler.move(b, a);
        m_assembler.move(tempRegister, b);
#else
#error Missing implementationg for matching swapping argument registers.
#endif
    }

    void prepareAndCall()
    {
        ASSERT(m_functionAddress.taggedPtr());
        ASSERT(!m_firstArgument || (m_firstArgument && !m_secondArgument) || (m_firstArgument && m_secondArgument));

        saveAllocatedCallerSavedRegisters();
        m_stackAllocator.alignStackPreFunctionCall();

        if (m_argumentCount == 2) {
            RELEASE_ASSERT(RegisterAllocator::isValidRegister(m_firstArgument));
            RELEASE_ASSERT(RegisterAllocator::isValidRegister(m_secondArgument));

            if (m_firstArgument != JSC::GPRInfo::argumentGPR0) {
                // If firstArgument is not in argumentGPR0, we need to handle potential conflicts:
                // -if secondArgument and firstArgument are in inversted registers, just swap the values.
                // -if secondArgument is in argumentGPR0 but firstArgument is not taking argumentGPR1, we can move in order secondArgument->argumentGPR1, firstArgument->argumentGPR0
                // -if secondArgument does not take argumentGPR0, firstArgument and secondArgument can be moved safely to destination.
                if (m_secondArgument == JSC::GPRInfo::argumentGPR0) {
                    if (m_firstArgument == JSC::GPRInfo::argumentGPR1)
                        swapArguments();
                    else {
                        m_assembler.move(JSC::GPRInfo::argumentGPR0, JSC::GPRInfo::argumentGPR1);
                        m_assembler.move(m_firstArgument, JSC::GPRInfo::argumentGPR0);
                    }
                } else {
                    m_assembler.move(m_firstArgument, JSC::GPRInfo::argumentGPR0);
                    if (m_secondArgument != JSC::GPRInfo::argumentGPR1)
                        m_assembler.move(m_secondArgument, JSC::GPRInfo::argumentGPR1);
                }
            } else {
                // We know firstArgument is already in place, we can safely move secondArgument.
                if (m_secondArgument != JSC::GPRInfo::argumentGPR1)
                    m_assembler.move(m_secondArgument, JSC::GPRInfo::argumentGPR1);
            }
        } else if (m_argumentCount == 1) {
            RELEASE_ASSERT(RegisterAllocator::isValidRegister(m_firstArgument));
            if (m_firstArgument != JSC::GPRInfo::argumentGPR0)
                m_assembler.move(m_firstArgument, JSC::GPRInfo::argumentGPR0);
        }

        JSC::MacroAssembler::Call call = m_assembler.call(JSC::OperationPtrTag);
        m_callRegistry.append(std::make_pair(call, m_functionAddress));
    }

    void cleanupPostCall()
    {
        m_stackAllocator.unalignStackPostFunctionCall();
        restoreAllocatedCallerSavedRegisters();
    }

    void saveAllocatedCallerSavedRegisters()
    {
        ASSERT(m_savedRegisterStackReferences.isEmpty());
        ASSERT(m_savedRegisters.isEmpty());
        const RegisterVector& allocatedRegisters = m_registerAllocator.allocatedRegisters();
        for (auto registerID : allocatedRegisters) {
            if (RegisterAllocator::isCallerSavedRegister(registerID))
                m_savedRegisters.append(registerID);
        }
        m_savedRegisterStackReferences = m_stackAllocator.push(m_savedRegisters);
    }

    void restoreAllocatedCallerSavedRegisters()
    {
        m_stackAllocator.pop(m_savedRegisterStackReferences, m_savedRegisters);
        m_savedRegisterStackReferences.clear();
    }

    JSC::MacroAssembler& m_assembler;
    RegisterAllocator& m_registerAllocator;
    StackAllocator& m_stackAllocator;
    Vector<std::pair<JSC::MacroAssembler::Call, CodePtr<JSC::OperationPtrTag>>, 32>& m_callRegistry;

    RegisterVector m_savedRegisters;
    StackAllocator::StackReferenceVector m_savedRegisterStackReferences;
    
    CodePtr<JSC::OperationPtrTag> m_functionAddress;
    unsigned m_argumentCount;
    JSC::MacroAssembler::RegisterID m_firstArgument;
    JSC::MacroAssembler::RegisterID m_secondArgument;
};

} // namespace WebCore

#endif // ENABLE(CSS_SELECTOR_JIT)

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 9, 2025.
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
#include <JavaScriptCore/MacroAssembler.h>
#include <limits>

namespace WebCore {

class StackAllocator {
public:
    class StackReference {
    public:
        StackReference()
            : m_offsetFromTop(std::numeric_limits<unsigned>::max())
        { }
        explicit StackReference(unsigned offset)
            : m_offsetFromTop(offset)
        { }
        operator unsigned() const { return m_offsetFromTop; }
        bool isValid() const { return m_offsetFromTop != std::numeric_limits<unsigned>::max(); }
    private:
        unsigned m_offsetFromTop;
    };

    typedef Vector<StackReference, maximumRegisterCount> StackReferenceVector;

    StackAllocator(JSC::MacroAssembler& assembler)
        : m_assembler(assembler)
        , m_offsetFromTop(0)
        , m_hasFunctionCallPadding(false)
    {
    }

    StackAllocator(const StackAllocator&) = default;

    StackReference stackTop()
    {
        return StackReference(m_offsetFromTop + stackUnitInBytes());
    }

    ~StackAllocator()
    {
        RELEASE_ASSERT(!m_offsetFromTop);
        RELEASE_ASSERT(!m_hasFunctionCallPadding);
    }

    StackReference allocateUninitialized()
    {
        return allocateUninitialized(1)[0];
    }

    StackReferenceVector allocateUninitialized(unsigned count)
    {
        RELEASE_ASSERT(!m_hasFunctionCallPadding);
        StackReferenceVector stackReferences;
        unsigned oldOffsetFromTop = m_offsetFromTop;
#if CPU(ARM64)
        for (unsigned i = 0; i < count - 1; i += 2) {
            m_offsetFromTop += stackUnitInBytes();
            stackReferences.append(StackReference(m_offsetFromTop - stackUnitInBytes() / 2));
            stackReferences.append(StackReference(m_offsetFromTop));
        }
        if (count % 2) {
            m_offsetFromTop += stackUnitInBytes();
            stackReferences.append(StackReference(m_offsetFromTop));
        }
#else
        for (unsigned i = 0; i < count; ++i) {
            m_offsetFromTop += stackUnitInBytes();
            stackReferences.append(StackReference(m_offsetFromTop));
        }
#endif
        m_assembler.addPtrNoFlags(JSC::MacroAssembler::TrustedImm32(-(m_offsetFromTop - oldOffsetFromTop)), JSC::MacroAssembler::stackPointerRegister);
        return stackReferences;
    }

    template<size_t inlineCapacity, typename OverflowHandler>
    StackReferenceVector push(const Vector<JSC::MacroAssembler::RegisterID, inlineCapacity, OverflowHandler>& registerIDs)
    {
        RELEASE_ASSERT(!m_hasFunctionCallPadding);

        StackReferenceVector stackReferences;

        if (registerIDs.isEmpty())
            return stackReferences;

#if CPU(ARM64)
        unsigned pushRegisterCount = registerIDs.size();
        for (unsigned i = 0; i < pushRegisterCount - 1; i += 2) {
            m_assembler.pushPair(registerIDs[i + 1], registerIDs[i]);
            m_offsetFromTop += stackUnitInBytes();
            stackReferences.append(StackReference(m_offsetFromTop - stackUnitInBytes() / 2));
            stackReferences.append(StackReference(m_offsetFromTop));
        }
        if (pushRegisterCount % 2)
            stackReferences.append(push(registerIDs[pushRegisterCount - 1]));
#else
        for (auto registerID : registerIDs)
            stackReferences.append(push(registerID));
#endif
        return stackReferences;
    }

    StackReference push(JSC::MacroAssembler::RegisterID registerID)
    {
        RELEASE_ASSERT(!m_hasFunctionCallPadding);
        m_assembler.pushToSave(registerID);
        m_offsetFromTop += stackUnitInBytes();
        return StackReference(m_offsetFromTop);
    }

    template<size_t inlineCapacity, typename OverflowHandler>
    void pop(const StackReferenceVector& stackReferences, const Vector<JSC::MacroAssembler::RegisterID, inlineCapacity, OverflowHandler>& registerIDs)
    {
        RELEASE_ASSERT(!m_hasFunctionCallPadding);

        unsigned popRegisterCount = registerIDs.size();
        RELEASE_ASSERT(stackReferences.size() == popRegisterCount);
#if CPU(ARM64)
        ASSERT(m_offsetFromTop >= stackUnitInBytes() * ((popRegisterCount + 1) / 2));
        unsigned popRegisterCountOdd = popRegisterCount % 2;
        if (popRegisterCountOdd)
            pop(stackReferences[popRegisterCount - 1], registerIDs[popRegisterCount - 1]);
        for (unsigned i = popRegisterCount - popRegisterCountOdd; i > 0; i -= 2) {
            RELEASE_ASSERT(stackReferences[i - 1] == m_offsetFromTop);
            RELEASE_ASSERT(stackReferences[i - 2] == m_offsetFromTop - stackUnitInBytes() / 2);
            RELEASE_ASSERT(m_offsetFromTop >= stackUnitInBytes());
            m_offsetFromTop -= stackUnitInBytes();
            m_assembler.popPair(registerIDs[i - 1], registerIDs[i - 2]);
        }
#else
        ASSERT(m_offsetFromTop >= stackUnitInBytes() * popRegisterCount);
        for (unsigned i = popRegisterCount; i > 0; --i)
            pop(stackReferences[i - 1], registerIDs[i - 1]);
#endif
    }

    void pop(StackReference stackReference, JSC::MacroAssembler::RegisterID registerID)
    {
        RELEASE_ASSERT(stackReference == m_offsetFromTop);
        RELEASE_ASSERT(!m_hasFunctionCallPadding);
        RELEASE_ASSERT(m_offsetFromTop >= stackUnitInBytes());
        m_offsetFromTop -= stackUnitInBytes();
        m_assembler.popToRestore(registerID);
    }

    void popAndDiscardUpTo(StackReference stackReference)
    {
        unsigned positionBeforeStackReference = stackReference - stackUnitInBytes();
        RELEASE_ASSERT(positionBeforeStackReference < m_offsetFromTop);

        unsigned stackDelta = m_offsetFromTop - positionBeforeStackReference;
        m_assembler.addPtr(JSC::MacroAssembler::TrustedImm32(stackDelta), JSC::MacroAssembler::stackPointerRegister);
        m_offsetFromTop -= stackDelta;
    }

    void alignStackPreFunctionCall()
    {
#if CPU(X86_64)
        RELEASE_ASSERT(!m_hasFunctionCallPadding);
        unsigned topAlignment = stackUnitInBytes();
        if ((topAlignment + m_offsetFromTop) % 16) {
            m_hasFunctionCallPadding = true;
            m_assembler.addPtrNoFlags(JSC::MacroAssembler::TrustedImm32(-stackUnitInBytes()), JSC::MacroAssembler::stackPointerRegister);
        }
#endif
    }

    void unalignStackPostFunctionCall()
    {
#if CPU(X86_64)
        if (m_hasFunctionCallPadding) {
            m_assembler.addPtrNoFlags(JSC::MacroAssembler::TrustedImm32(stackUnitInBytes()), JSC::MacroAssembler::stackPointerRegister);
            m_hasFunctionCallPadding = false;
        }
#endif
    }

    void merge(StackAllocator&& stackA, StackAllocator&& stackB)
    {
        RELEASE_ASSERT(stackA.m_offsetFromTop == stackB.m_offsetFromTop);
        RELEASE_ASSERT(stackA.m_hasFunctionCallPadding == stackB.m_hasFunctionCallPadding);
        ASSERT(&stackA.m_assembler == &stackB.m_assembler);
        ASSERT(&m_assembler == &stackB.m_assembler);

        m_offsetFromTop = stackA.m_offsetFromTop;
        m_hasFunctionCallPadding = stackA.m_hasFunctionCallPadding;

        stackA.reset();
        stackB.reset();
    }

    void merge(StackAllocator&& stackA, StackAllocator&& stackB, StackAllocator&& stackC)
    {
        RELEASE_ASSERT(stackA.m_offsetFromTop == stackB.m_offsetFromTop);
        RELEASE_ASSERT(stackA.m_offsetFromTop == stackC.m_offsetFromTop);
        RELEASE_ASSERT(stackA.m_hasFunctionCallPadding == stackB.m_hasFunctionCallPadding);
        RELEASE_ASSERT(stackA.m_hasFunctionCallPadding == stackC.m_hasFunctionCallPadding);
        ASSERT(&stackA.m_assembler == &stackB.m_assembler);
        ASSERT(&stackA.m_assembler == &stackC.m_assembler);
        ASSERT(&m_assembler == &stackB.m_assembler);

        m_offsetFromTop = stackA.m_offsetFromTop;
        m_hasFunctionCallPadding = stackA.m_hasFunctionCallPadding;

        stackA.reset();
        stackB.reset();
        stackC.reset();
    }

    JSC::MacroAssembler::Address addressOf(StackReference stackReference)
    {
        return JSC::MacroAssembler::Address(JSC::MacroAssembler::stackPointerRegister, offsetToStackReference(stackReference));
    }

    StackAllocator& operator=(const StackAllocator& other)
    {
        RELEASE_ASSERT(&m_assembler == &other.m_assembler);
        m_offsetFromTop = other.m_offsetFromTop;
        m_hasFunctionCallPadding = other.m_hasFunctionCallPadding;
        return *this;
    }


private:
    static unsigned stackUnitInBytes()
    {
        return JSC::MacroAssembler::pushToSaveByteOffset();
    }

    unsigned offsetToStackReference(StackReference stackReference)
    {
        RELEASE_ASSERT(m_offsetFromTop >= stackReference);
        return m_offsetFromTop - stackReference;
    }

    void reset()
    {
        m_offsetFromTop = 0;
        m_hasFunctionCallPadding = false;
    }

    JSC::MacroAssembler& m_assembler;
    unsigned m_offsetFromTop;
    bool m_hasFunctionCallPadding;
};

} // namespace WebCore

#endif // ENABLE(CSS_SELECTOR_JIT)

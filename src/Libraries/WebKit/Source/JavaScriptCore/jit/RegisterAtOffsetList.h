/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 14, 2022.
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

#if ENABLE(ASSEMBLER)

#include "RegisterAtOffset.h"
#include "RegisterSet.h"
#include <wtf/FixedVector.h>

namespace JSC {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(RegisterAtOffsetList);
class RegisterAtOffsetList {
    WTF_MAKE_STRUCT_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(RegisterAtOffsetList);
public:
    enum OffsetBaseType { FramePointerBased, ZeroBased };

    RegisterAtOffsetList();
    explicit RegisterAtOffsetList(RegisterSet, OffsetBaseType = FramePointerBased);

    void dump(PrintStream&) const;

    size_t registerCount() const { return m_registers.size(); }
    size_t sizeOfAreaInBytes() const { return m_sizeOfAreaInBytes; }

    const RegisterAtOffset& at(size_t index) const
    {
        return m_registers.at(index);
    }

    void adjustOffsets(ptrdiff_t addend)
    {
        // This preserves m_sizeOfAreaInBytes
        for (RegisterAtOffset &item : m_registers)
            item = RegisterAtOffset { item.reg(), item.offset() + addend, item.width() };
    }

    RegisterAtOffset* find(Reg) const;
    unsigned indexOf(Reg) const; // Returns UINT_MAX if not found.

    FixedVector<RegisterAtOffset>::const_iterator begin() const { return m_registers.begin(); }
    FixedVector<RegisterAtOffset>::const_iterator end() const { return m_registers.end(); }


    static const RegisterAtOffsetList& llintBaselineCalleeSaveRegisters(); // Registers and Offsets saved and used by the LLInt.
    static const RegisterAtOffsetList& dfgCalleeSaveRegisters(); // Registers and Offsets saved and used by DFG.

private:
    FixedVector<RegisterAtOffset> m_registers;
    size_t m_sizeOfAreaInBytes { 0 };
};

} // namespace JSC

#endif // ENABLE(ASSEMBLER)

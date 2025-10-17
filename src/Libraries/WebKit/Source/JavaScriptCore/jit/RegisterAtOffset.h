/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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

#include "Reg.h"
#include "Width.h"
#include <cstddef>
#include <wtf/PrintStream.h>

namespace JSC {

class RegisterAtOffset {
public:
    RegisterAtOffset() = default;
    
    RegisterAtOffset(Reg reg, ptrdiff_t offset, Width width)
        : m_regIndex(reg.index())
        , m_width(width == Width128)
        , m_offsetBits((offset >> 2) & 0xFFFFFFFFFFFFFF)
    {
        ASSERT(!(offset & 0b11));
        ASSERT(width == conservativeWidthWithoutVectors(reg) || Options::useWasmSIMD());
        ASSERT(reg.index() < (1 << 6));
        ASSERT(Reg::last().index() < (1 << 6));
        ASSERT(this->reg() == reg);
        ASSERT(this->offset() == offset);
        ASSERT(this->width() == width);
    }
    
    bool operator!() const { return !reg(); }
    
    Reg reg() const { return Reg::fromIndex(m_regIndex); }
    ptrdiff_t offset() const { return m_offsetBits << 2; }
    size_t byteSize() const { return bytesForWidth(width()); }
    Width width() const { return m_width ? conservativeWidth(reg()) : conservativeWidthWithoutVectors(reg()); }
    int offsetAsIndex() const { ASSERT(!(offset() % sizeof(CPURegister))); return offset() / static_cast<int>(sizeof(CPURegister)); }

    bool operator==(const RegisterAtOffset& other) const
    {
        return reg() == other.reg() && offset() == other.offset() && width() == other.width();
    }
    
    bool operator<(const RegisterAtOffset& other) const
    {
        if (reg() != other.reg())
            return reg() < other.reg();
        return offset() < other.offset();
    }
    
    static Reg getReg(RegisterAtOffset* value) { return value->reg(); }
    
    void dump(PrintStream& out) const;

private:
    unsigned m_regIndex : 7 { Reg().index() };
    unsigned m_width : 1 { false };
    ptrdiff_t m_offsetBits : (sizeof(ptrdiff_t) * CHAR_BIT - 7 - 1) { 0 };
};

} // namespace JSC

#endif // ENABLE(ASSEMBLER)

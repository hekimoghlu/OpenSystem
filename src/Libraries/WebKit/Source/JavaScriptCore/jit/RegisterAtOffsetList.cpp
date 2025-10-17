/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 29, 2025.
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
#include "RegisterAtOffsetList.h"

#if ENABLE(ASSEMBLER)

#include <wtf/ListDump.h>

namespace JSC {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(RegisterAtOffsetList);

RegisterAtOffsetList::RegisterAtOffsetList() { }

RegisterAtOffsetList::RegisterAtOffsetList(RegisterSet registerSetBuilder, OffsetBaseType offsetBaseType)
    : m_registers(registerSetBuilder.numberOfSetRegisters())
{
    ASSERT(!registerSetBuilder.hasAnyWideRegisters() || Options::useWasmSIMD());

    size_t sizeOfAreaInBytes = registerSetBuilder.byteSizeOfSetRegisters();
    m_sizeOfAreaInBytes = sizeOfAreaInBytes;
#if USE(JSVALUE64)
    static_assert(sizeof(CPURegister) == sizeof(double));
    ASSERT(this->sizeOfAreaInBytes() == registerCount() * sizeof(CPURegister) || Options::useWasmSIMD());
#endif    

    ptrdiff_t startOffset = 0;
    if (offsetBaseType == FramePointerBased)
        startOffset = -static_cast<ptrdiff_t>(sizeOfAreaInBytes);

    ptrdiff_t offset = startOffset;
    unsigned index = 0;

    registerSetBuilder.forEachWithWidth([&] (Reg reg, Width width) {
        offset = WTF::roundUpToMultipleOf(alignmentForWidth(width), offset);
        m_registers[index++] = RegisterAtOffset(reg, offset, width);
        offset += bytesForWidth(width);
    });

    ASSERT(static_cast<size_t>(offset - startOffset) == sizeOfAreaInBytes);
}

void RegisterAtOffsetList::dump(PrintStream& out) const
{
    out.print(listDump(m_registers));
}

RegisterAtOffset* RegisterAtOffsetList::find(Reg reg) const
{
    return tryBinarySearch<RegisterAtOffset, Reg>(m_registers, m_registers.size(), reg, RegisterAtOffset::getReg);
}

unsigned RegisterAtOffsetList::indexOf(Reg reg) const
{
    if (RegisterAtOffset* pointer = find(reg))
        return pointer - m_registers.begin();
    return UINT_MAX;
}

const RegisterAtOffsetList& RegisterAtOffsetList::llintBaselineCalleeSaveRegisters()
{
    static std::once_flag onceKey;
    static LazyNeverDestroyed<RegisterAtOffsetList> result;
    std::call_once(onceKey, [] {
        result.construct(RegisterSetBuilder::llintBaselineCalleeSaveRegisters());
    });
    return result.get();
}

const RegisterAtOffsetList& RegisterAtOffsetList::dfgCalleeSaveRegisters()
{
    static std::once_flag onceKey;
    static LazyNeverDestroyed<RegisterAtOffsetList> result;
    std::call_once(onceKey, [] {
        result.construct(RegisterSetBuilder::dfgCalleeSaveRegisters());
    });
    return result.get();
}

} // namespace JSC

#endif // ENABLE(ASSEMBLER)


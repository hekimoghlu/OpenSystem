/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 11, 2024.
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
#include "B3WasmBoundsCheckValue.h"
#include "WasmMemory.h"

#if ENABLE(B3_JIT)

#include "B3ValueInlines.h"

namespace JSC { namespace B3 {

WasmBoundsCheckValue::~WasmBoundsCheckValue() = default;

WasmBoundsCheckValue::WasmBoundsCheckValue(Origin origin, GPRReg pinnedSize, Value* ptr, unsigned offset)
    : Value(CheckedOpcode, WasmBoundsCheck, One, origin, ptr)
    , m_offset(offset)
    , m_boundsType(Type::Pinned)
{
    m_bounds.pinnedSize = pinnedSize;
}

WasmBoundsCheckValue::WasmBoundsCheckValue(Origin origin, Value* ptr, unsigned offset, size_t maximum)
    : Value(CheckedOpcode, WasmBoundsCheck, One, origin, ptr)
    , m_offset(offset)
    , m_boundsType(Type::Maximum)
{
    size_t redzoneLimit = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + Wasm::Memory::fastMappedRedzoneBytes();
    ASSERT_UNUSED(redzoneLimit, maximum <= redzoneLimit);
    m_bounds.maximum = maximum;
}

void WasmBoundsCheckValue::dumpMeta(CommaPrinter& comma, PrintStream& out) const
{
    switch (m_boundsType) {
    case Type::Pinned:
        out.print(comma, "offset = ", m_offset, comma, "pinnedSize = ", m_bounds.pinnedSize);
        break;
    case Type::Maximum:
        out.print(comma, "offset = ", m_offset, comma, "maximum = ", m_bounds.maximum);
        break;
    }
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

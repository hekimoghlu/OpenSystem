/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 25, 2024.
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

#include "CacheableIdentifier.h"
#include "DFGRegisteredStructure.h"
#include "HeapCell.h"
#include "PrivateFieldPutKind.h"
#include <wtf/OptionSet.h>
#include <wtf/StdLibExtras.h>

#if ENABLE(DFG_JIT)

namespace JSC { namespace DFG {

// This type used in passing an immediate argument to Node constructor;
// distinguishes an immediate value (typically an index into a CodeBlock data structure - 
// a constant index, argument, or identifier) from a Node*.
struct OpInfo {
    OpInfo() : m_value(0) { }
    template<
        typename IntegralType,
        typename Constraint = typename std::enable_if<(std::is_integral<IntegralType>::value || std::is_enum<IntegralType>::value) && sizeof(IntegralType) <= sizeof(uint64_t)>::type>
    explicit OpInfo(IntegralType value)
        : m_value(static_cast<uint64_t>(value)) { }
    explicit OpInfo(RegisteredStructure structure) : m_value(static_cast<uint64_t>(std::bit_cast<uintptr_t>(structure))) { }
    explicit OpInfo(Operand op) : m_value(op.asBits()) { }
    explicit OpInfo(CacheableIdentifier identifier) : m_value(static_cast<uint64_t>(identifier.rawBits())) { }
    explicit OpInfo(ECMAMode ecmaMode) : m_value(ecmaMode.value()) { }
    explicit OpInfo(PrivateFieldPutKind putKind) : m_value(putKind.value()) { }
    template<typename EnumType>
    explicit OpInfo(OptionSet<EnumType> optionSet) : m_value(optionSet.toRaw()) { }

    template <typename T>
    explicit OpInfo(T* ptr)
    {
        static_assert(!std::is_base_of<HeapCell, T>::value, "To make an OpInfo with a cell in it, the cell must be registered or frozen.");
        static_assert(!std::is_base_of<StructureSet, T>::value, "To make an OpInfo with a structure set in, make sure to use RegisteredStructureSet.");
        m_value = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
    }

    uint64_t m_value;
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

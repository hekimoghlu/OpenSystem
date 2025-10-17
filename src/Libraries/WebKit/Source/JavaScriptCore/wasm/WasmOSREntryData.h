/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 30, 2022.
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

#if ENABLE(WEBASSEMBLY_OMGJIT) || ENABLE(WEBASSEMBLY_BBQJIT)

#include "B3Type.h"
#include "B3ValueRep.h"
#include "WasmFormat.h"
#include <wtf/FixedVector.h>
#include <wtf/TZoneMalloc.h>

namespace JSC { namespace Wasm {

class OSREntryValue final : public B3::ValueRep {
public:
    OSREntryValue() = default;
    OSREntryValue(const B3::ValueRep& valueRep, B3::Type type)
        : B3::ValueRep(valueRep)
        , m_type(type)
    {
    }

    B3::Type type() const { return m_type; }

private:
    B3::Type m_type { };
};

class OSREntryData {
    WTF_MAKE_NONCOPYABLE(OSREntryData);
    WTF_MAKE_TZONE_ALLOCATED(OSREntryData);
public:
    OSREntryData(FunctionCodeIndex functionIndex, uint32_t loopIndex, StackMap&& stackMap)
        : m_functionIndex(functionIndex)
        , m_loopIndex(loopIndex)
        , m_values(WTFMove(stackMap))
    {
    }

    FunctionCodeIndex functionIndex() const { return m_functionIndex; }
    uint32_t loopIndex() const { return m_loopIndex; }
    const StackMap& values() { return m_values; }

private:
    FunctionCodeIndex m_functionIndex;
    uint32_t m_loopIndex;
    StackMap m_values;
};

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY_OMGJIT)

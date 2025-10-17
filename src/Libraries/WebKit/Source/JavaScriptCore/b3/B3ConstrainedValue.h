/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 6, 2025.
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

#include "B3ValueRep.h"
#include "B3Width.h"
#include "WasmCallingConvention.h"

namespace JSC { namespace B3 {

class Value;

#if ENABLE(WEBASSEMBLY)
struct ArgumentLocation {
    ArgumentLocation(Wasm::ValueLocation loc, Width width)
        : location(loc)
        , width(width)
    {
    }

    ArgumentLocation() { }

    Wasm::ValueLocation location;
    Width width;
};
#endif

class ConstrainedValue {
public:
    ConstrainedValue()
    {
    }

    ConstrainedValue(Value* value)
        : m_value(value)
        , m_rep(ValueRep::WarmAny)
    {
    }

    ConstrainedValue(Value* value, const ValueRep& rep)
        : m_value(value)
        , m_rep(rep)
    {
    }

#if ENABLE(WEBASSEMBLY)
#if USE(JSVALUE32_64)
    ConstrainedValue(Value* value, const Wasm::ArgumentLocation& loc)
        : m_value(value)
    {
        if (loc.location.isGPR() && loc.usedWidth == Width32)
            m_rep = B3::ValueRep(loc.location.jsr().payloadGPR());
        else
            m_rep = B3::ValueRep(loc.location);
    }
#else
    ConstrainedValue(Value* value, const Wasm::ArgumentLocation& loc)
        : m_value(value)
        , m_rep(loc.location)
    {
    }
#endif // USE(JSVALUE32_64)
#endif // ENABLE(WEBASSEMBLY)

    explicit operator bool() const { return m_value || m_rep; }

    Value* value() const { return m_value; }
    const ValueRep& rep() const { return m_rep; }

    void dump(PrintStream& out) const;

private:
    Value* m_value;
    ValueRep m_rep;
};

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

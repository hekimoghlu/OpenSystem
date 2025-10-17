/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 16, 2024.
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

#include "ResultType.h"
#include <wtf/Packed.h>

namespace JSC {

class SnippetOperand {
    enum ConstOrVarType : uint8_t {
        Variable,
        ConstInt32,
        ConstDouble
    };

public:
    SnippetOperand()
        : m_resultType(ResultType::unknownType())
    { }

    SnippetOperand(ResultType resultType)
        : m_resultType(resultType)
    { }

    bool mightBeNumber() const { return m_resultType.mightBeNumber(); }
    bool definitelyIsNumber() const { return m_resultType.definitelyIsNumber(); }

    bool definitelyIsBoolean() const { return m_resultType.definitelyIsBoolean(); }

    bool isConst() const { return m_type != Variable; }
    bool isConstInt32() const { return m_type == ConstInt32; }
    bool isConstDouble() const { return m_type == ConstDouble; }
    bool isPositiveConstInt32() const { return isConstInt32() && asConstInt32() > 0; }

    int64_t asRawBits() const { return m_val.get().rawBits; }

    int32_t asConstInt32() const
    {
        ASSERT(m_type == ConstInt32);
        return m_val.get().int32Val;
    }

    double asConstDouble() const
    {
        ASSERT(m_type == ConstDouble);
        return m_val.get().doubleVal;
    }

    double asConstNumber() const
    {
        if (isConstInt32())
            return asConstInt32();
        ASSERT(isConstDouble());
        return asConstDouble();
    }

    void setConstInt32(int32_t value)
    {
        m_type = ConstInt32;
        UnionType u;
        u.int32Val = value;
        m_val = WTFMove(u);
    }

    void setConstDouble(double value)
    {
        m_type = ConstDouble;
        UnionType u;
        u.doubleVal = value;
        m_val = WTFMove(u);
    }

private:
    ResultType m_resultType;
    ConstOrVarType m_type { Variable };
    union UnionType {
        int32_t int32Val;
        double doubleVal;
        int64_t rawBits;
    };
    Packed<UnionType> m_val;
};
static_assert(alignof(SnippetOperand) == 1);

} // namespace JSC

#endif // ENABLE(JIT)

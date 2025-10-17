/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 23, 2022.
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

#include "B3Type.h"
#include <wtf/PrintStream.h>

#if !ASSERT_ENABLED
IGNORE_RETURN_TYPE_WARNINGS_BEGIN
#endif

namespace JSC { namespace B3 {

template<typename T>
class TypeMap {
public:
    TypeMap() = default;
    
    T& at(Type type)
    {
        switch (type.kind()) {
        case Void:
            return m_void;
        case Int32:
            return m_int32;
        case Int64:
            return m_int64;
        case Float:
            return m_float;
        case Double:
            return m_double;
        case Tuple:
            return m_tuple;
        case V128:
            return m_vector;
        }
        ASSERT_NOT_REACHED();
    }
    
    const T& at(Type type) const
    {
        return std::bit_cast<TypeMap*>(this)->at(type);
    }
    
    T& operator[](Type type)
    {
        return at(type);
    }
    
    const T& operator[](Type type) const
    {
        return at(type);
    }
    
    void dump(PrintStream& out) const
    {
        out.print(
            "{void = ", m_void,
            ", int32 = ", m_int32,
            ", int64 = ", m_int64,
            ", float = ", m_float,
            ", double = ", m_double,
            ", vector = ", m_vector,
            ", tuple = ", m_tuple, "}");
    }
    
private:
    T m_void { };
    T m_int32 { };
    T m_int64 { };
    T m_float { };
    T m_double { };
    T m_vector { };
    T m_tuple { };
};

} } // namespace JSC::B3

#if !ASSERT_ENABLED
IGNORE_RETURN_TYPE_WARNINGS_END
#endif

#endif // ENABLE(B3_JIT)

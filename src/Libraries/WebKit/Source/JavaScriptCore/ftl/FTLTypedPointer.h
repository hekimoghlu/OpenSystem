/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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

#if ENABLE(FTL_JIT)

#include "FTLAbbreviatedTypes.h"
#include "FTLAbstractHeap.h"

namespace JSC { namespace FTL {

class TypedPointer {
public:
    TypedPointer()
        : m_heap(nullptr)
        , m_value(nullptr)
    {
    }
    
    TypedPointer(const AbstractHeap& heap, LValue value)
        : m_heap(&heap)
        , m_value(value)
    {
    }
    
    explicit operator bool() const
    {
        ASSERT(!m_heap == !m_value);
        return !!m_heap;
    }
    
    const AbstractHeap* heap() const { return m_heap; }
    LValue value() const { return m_value; }

private:
    const AbstractHeap* m_heap;
    LValue m_value;
};

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)

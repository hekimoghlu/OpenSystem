/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 5, 2024.
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

#include "AirCode.h"
#include "B3BasicBlock.h"
#include "B3Procedure.h"
#include "B3Value.h"

namespace JSC { namespace B3 {
    
template<typename ValueType, typename... Arguments>
ValueType* Procedure::add(Arguments... arguments)
{
    return static_cast<ValueType*>(addValueImpl(Value::allocate<ValueType>(arguments...)));
}

inline Type Procedure::extractFromTuple(Type tuple, unsigned index) const
{
    ASSERT(tuple.tupleIndex() < m_tuples.size());
    ASSERT(index < m_tuples[tuple.tupleIndex()].size());
    return m_tuples[tuple.tupleIndex()][index];
}

inline SparseCollection<Air::StackSlot>& Procedure::stackSlots() { return m_code->stackSlots(); }
inline const SparseCollection<Air::StackSlot>& Procedure::stackSlots() const { return m_code->stackSlots(); }

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

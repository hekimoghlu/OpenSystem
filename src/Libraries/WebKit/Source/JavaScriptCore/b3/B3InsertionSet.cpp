/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 13, 2023.
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
#include "B3InsertionSet.h"

#if ENABLE(B3_JIT)

#include "B3BasicBlock.h"
#include "B3Procedure.h"
#include "B3Value.h"
#include <wtf/BubbleSort.h>

namespace JSC { namespace B3 {

Value* InsertionSet::insertIntConstant(size_t index, Origin origin, Type type, int64_t value)
{
    return insertValue(index, m_procedure.addIntConstant(origin, type, value));
}

Value* InsertionSet::insertIntConstant(size_t index, Value* likeValue, int64_t value)
{
    return insertIntConstant(index, likeValue->origin(), likeValue->type(), value);
}

Value* InsertionSet::insertBottom(size_t index, Origin origin, Type type)
{
    if (type.isTuple())
        return insertValue(index, m_procedure.addBottom(origin, type));
    Value*& bottom = m_bottomForType[type];
    if (!bottom)
        bottom = insertValue(index, m_procedure.addBottom(origin, type));
    return bottom;
}

Value* InsertionSet::insertBottom(size_t index, Value* likeValue)
{
    return insertBottom(index, likeValue->origin(), likeValue->type());
}

Value* InsertionSet::insertClone(size_t index, Value* value)
{
    return insertValue(index, m_procedure.clone(value));
}

void InsertionSet::execute(BasicBlock* block)
{
    for (Insertion& insertion : m_insertions)
        insertion.element()->owner = block;
    bubbleSort(m_insertions.begin(), m_insertions.end());
    executeInsertions(block->m_values, m_insertions);
    m_bottomForType = TypeMap<Value*>();
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)


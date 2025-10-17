/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 22, 2025.
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

#include "B3BasicBlock.h"
#include "B3ProcedureInlines.h"
#include "B3Value.h"

namespace JSC { namespace B3 {

template<typename ValueType, typename... Arguments>
ValueType* BasicBlock::appendNew(Procedure& procedure, Arguments... arguments)
{
    ValueType* result = procedure.add<ValueType>(arguments...);
    append(result);
    return result;
}

template<typename ValueType, typename... Arguments>
ValueType* BasicBlock::replaceLastWithNew(Procedure& procedure, Arguments... arguments)
{
    ValueType* result = procedure.add<ValueType>(arguments...);
    replaceLast(procedure, result);
    return result;
}

inline const FrequentedBlock& BasicBlock::taken() const
{
    ASSERT(last()->opcode() == Jump || last()->opcode() == Branch);
    return m_successors[0];
}

inline FrequentedBlock& BasicBlock::taken()
{
    ASSERT(last()->opcode() == Jump || last()->opcode() == Branch);
    return m_successors[0];
}

inline const FrequentedBlock& BasicBlock::notTaken() const
{
    ASSERT(last()->opcode() == Branch);
    return m_successors[1];
}

inline FrequentedBlock& BasicBlock::notTaken()
{
    ASSERT(last()->opcode() == Branch);
    return m_successors[1];
}

inline const FrequentedBlock& BasicBlock::fallThrough() const
{
    ASSERT(last()->opcode() == Branch || last()->opcode() == Switch);
    return m_successors.last();
}

inline FrequentedBlock& BasicBlock::fallThrough()
{
    ASSERT(last()->opcode() == Branch || last()->opcode() == Switch);
    return m_successors.last();
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

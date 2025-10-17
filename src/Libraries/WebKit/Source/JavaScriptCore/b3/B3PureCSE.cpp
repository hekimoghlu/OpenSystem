/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 16, 2022.
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
#include "B3PureCSE.h"

#if ENABLE(B3_JIT)

#include "B3Dominators.h"
#include "B3PhaseScope.h"
#include "B3Value.h"
#include "B3ValueInlines.h"

namespace JSC { namespace B3 {

PureCSE::PureCSE() = default;

PureCSE::~PureCSE() = default;

void PureCSE::clear()
{
    m_map.clear();
}

Value* PureCSE::findMatch(const ValueKey& key, BasicBlock* block, Dominators& dominators)
{
    if (!key)
        return nullptr;

    auto iter = m_map.find(key);
    if (iter == m_map.end())
        return nullptr;

    for (Value* match : iter->value) {
        // Value is invalidated.
        if (!match->owner)
            continue;
        // Value is moved to a new BasicBlock which is not inserted yet.
        // In that case, we should just ignore it. PureCSE will be recomputed after new BasicBlocks are actually inserted.
        if (!match->owner->isInserted())
            continue;
        if (dominators.dominates(match->owner, block))
            return match;
    }

    return nullptr;
}

bool PureCSE::process(Value* value, Dominators& dominators)
{
    if (value->opcode() == Identity || value->isConstant())
        return false;

    ValueKey key = value->key();
    if (!key)
        return false;

    Matches& matches = m_map.add(key, Matches()).iterator->value;

    for (Value* match : matches) {
        // Value is invalidated.
        if (!match->owner)
            continue;
        // Value is moved to a new BasicBlock which is not inserted yet.
        // In that case, we should just ignore it. PureCSE will be recomputed after new BasicBlocks are actually inserted.
        if (!match->owner->isInserted())
            continue;
        if (dominators.dominates(match->owner, value->owner)) {
            value->replaceWithIdentity(match);
            return true;
        }
    }

    matches.append(value);
    return false;
}

bool pureCSE(Procedure& proc)
{
    PhaseScope phaseScope(proc, "pureCSE"_s);
    
    Dominators& dominators = proc.dominators();
    PureCSE pureCSE;
    bool result = false;
    for (BasicBlock* block : proc.blocksInPreOrder()) {
        for (Value* value : *block) {
            result |= value->performSubstitution();
            result |= pureCSE.process(value, dominators);
        }
    }
    
    return result;
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)


/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 27, 2022.
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
#include "B3SwitchValue.h"

#if ENABLE(B3_JIT)

#include "B3BasicBlockInlines.h"
#include "B3ValueInlines.h"
#include <wtf/ListDump.h>

namespace JSC { namespace B3 {

SwitchValue::~SwitchValue() = default;

BasicBlock* SwitchValue::fallThrough(const BasicBlock* owner)
{
    ASSERT(hasFallThrough());
    BasicBlock* fallThrough = owner->successor(owner->numSuccessors() - 1).block();
    ASSERT(fallThrough == owner->fallThrough().block());
    return fallThrough;
}

bool SwitchValue::hasFallThrough(const BasicBlock* block) const
{
    unsigned numSuccessors = block->numSuccessors();
    unsigned numValues = m_values.size();
    RELEASE_ASSERT(numValues == numSuccessors || numValues + 1 == numSuccessors);
    
    return numValues + 1 == numSuccessors;
}

bool SwitchValue::hasFallThrough() const
{
    return hasFallThrough(owner);
}

void SwitchValue::setFallThrough(BasicBlock* block, const FrequentedBlock& target)
{
    if (!hasFallThrough())
        block->successors().append(target);
    else
        block->successors().last() = target;
    ASSERT(hasFallThrough(block));
}

void SwitchValue::appendCase(BasicBlock* block, const SwitchCase& switchCase)
{
    if (!hasFallThrough())
        block->successors().append(switchCase.target());
    else {
        block->successors().append(block->successors().last());
        block->successor(block->numSuccessors() - 2) = switchCase.target();
    }
    m_values.append(switchCase.caseValue());
}

void SwitchValue::setFallThrough(const FrequentedBlock& target)
{
    setFallThrough(owner, target);
}

void SwitchValue::appendCase(const SwitchCase& switchCase)
{
    appendCase(owner, switchCase);
}

void SwitchValue::dumpSuccessors(const BasicBlock* block, PrintStream& out) const
{
    // We must not crash due to a number-of-successors mismatch! Someone debugging a
    // number-of-successors bug will want to dump IR!
    if (numCaseValues() + 1 != block->numSuccessors()) {
        Value::dumpSuccessors(block, out);
        return;
    }
    
    out.print(cases(block));
}

void SwitchValue::dumpMeta(CommaPrinter& comma, PrintStream& out) const
{
    out.print(comma, "cases = [", listDump(m_values), "]");
}

SwitchValue::SwitchValue(Origin origin, Value* child)
    : Value(CheckedOpcode, Switch, Void, One, origin, child)
{
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

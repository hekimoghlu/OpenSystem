/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 22, 2024.
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
#include "AirBasicBlock.h"

#if ENABLE(B3_JIT)

#include "B3BasicBlockUtils.h"
#include <wtf/ListDump.h>
#include <wtf/TZoneMallocInlines.h>

namespace JSC { namespace B3 { namespace Air {

const char* const BasicBlock::dumpPrefix = "#";

WTF_MAKE_TZONE_ALLOCATED_IMPL(BasicBlock);

void BasicBlock::setSuccessors(FrequentedBlock target)
{
    m_successors.resize(1);
    m_successors[0] = target;
}

void BasicBlock::setSuccessors(FrequentedBlock taken, FrequentedBlock notTaken)
{
    m_successors.resize(2);
    m_successors[0] = taken;
    m_successors[1] = notTaken;
}

bool BasicBlock::addPredecessor(BasicBlock* block)
{
    return B3::addPredecessor(this, block);
}

bool BasicBlock::removePredecessor(BasicBlock* block)
{
    return B3::removePredecessor(this, block);
}

bool BasicBlock::replacePredecessor(BasicBlock* from, BasicBlock* to)
{
    return B3::replacePredecessor(this, from, to);
}

void BasicBlock::dump(PrintStream& out) const
{
    out.print(dumpPrefix, m_index);
}

void BasicBlock::deepDump(PrintStream& out) const
{
    dumpHeader(out);
    for (const Inst& inst : *this)
        out.print(tierName, "    ", inst, "\n");
    dumpFooter(out);
}

void BasicBlock::dumpHeader(PrintStream& out) const
{
    out.print(tierName, "BB", *this, ": ; frequency = ", m_frequency, "\n");
    if (predecessors().size())
        out.print(tierName, "  Predecessors: ", pointerListDump(predecessors()), "\n");
}

void BasicBlock::dumpFooter(PrintStream& out) const
{
    if (successors().size())
        out.print(tierName, "  Successors: ", listDump(successors()), "\n");
}

BasicBlock::BasicBlock(unsigned index, double frequency)
    : m_index(index)
    , m_frequency(frequency)
{
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)

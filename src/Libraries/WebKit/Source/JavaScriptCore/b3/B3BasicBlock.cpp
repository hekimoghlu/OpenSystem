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
#include "config.h"
#include "B3BasicBlock.h"

#if ENABLE(B3_JIT)

#include "B3BasicBlockInlines.h"
#include "B3BasicBlockUtils.h"
#include "B3Procedure.h"
#include "B3ValueInlines.h"
#include <wtf/ListDump.h>
#include <wtf/TZoneMallocInlines.h>

namespace JSC { namespace B3 {

const char* const BasicBlock::dumpPrefix = "#";

WTF_MAKE_TZONE_ALLOCATED_IMPL(BasicBlock);

BasicBlock::BasicBlock(unsigned index, double frequency)
    : m_index(index)
    , m_frequency(frequency)
{
}

BasicBlock::~BasicBlock() = default;

void BasicBlock::append(Value* value)
{
    m_values.append(value);
    value->owner = this;
}

void BasicBlock::appendNonTerminal(Value* value)
{
    m_values.append(m_values.last());
    m_values[m_values.size() - 2] = value;
    value->owner = this;
}

void BasicBlock::removeLast(Procedure& proc)
{
    ASSERT(!m_values.isEmpty());
    proc.deleteValue(m_values.takeLast());
}

void BasicBlock::replaceLast(Procedure& proc, Value* value)
{
    removeLast(proc);
    append(value);
}

Value* BasicBlock::appendIntConstant(Procedure& proc, Origin origin, Type type, int64_t value)
{
    Value* result = proc.addIntConstant(origin, type, value);
    append(result);
    return result;
}

Value* BasicBlock::appendIntConstant(Procedure& proc, Value* likeValue, int64_t value)
{
    return appendIntConstant(proc, likeValue->origin(), likeValue->type(), value);
}

Value* BasicBlock::appendBoolConstant(Procedure& proc, Origin origin, bool value)
{
    return appendIntConstant(proc, origin, Int32, value ? 1 : 0);
}

void BasicBlock::clearSuccessors()
{
    m_successors.clear();
}

void BasicBlock::appendSuccessor(FrequentedBlock target)
{
    m_successors.append(target);
}

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

bool BasicBlock::replaceSuccessor(BasicBlock* from, BasicBlock* to)
{
    bool result = false;
    for (BasicBlock*& successor : successorBlocks()) {
        if (successor == from) {
            successor = to;
            result = true;
            
            // Keep looping because a successor may be mentioned multiple times, like in a Switch.
        }
    }
    return result;
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

void BasicBlock::updatePredecessorsAfter()
{
    B3::updatePredecessorsAfter(this);
}

void BasicBlock::dump(PrintStream& out) const
{
    out.print(dumpPrefix, m_index);
}

void BasicBlock::deepDump(const Procedure& proc, PrintStream& out) const
{
    out.print(tierName, "BB", *this, ": ; frequency = ", m_frequency, "\n");
    if (predecessors().size())
        out.print(tierName, "  Predecessors: ", pointerListDump(predecessors()), "\n");
    for (Value* value : *this)
        out.print(tierName, "    ", B3::deepDump(proc, value), "\n");
    if (!successors().isEmpty()) {
        out.print(tierName, "  Successors: ");
        if (size())
            last()->dumpSuccessors(this, out);
        else
            out.print(listDump(successors()));
        out.print("\n");
    }
}

Value* BasicBlock::appendNewControlValue(Procedure& proc, Opcode opcode, Origin origin)
{
    RELEASE_ASSERT(opcode == Oops || opcode == Return);
    clearSuccessors();
    return appendNew<Value>(proc, opcode, origin);
}

Value* BasicBlock::appendNewControlValue(Procedure& proc, Opcode opcode, Origin origin, Value* value)
{
    RELEASE_ASSERT(opcode == Return);
    clearSuccessors();
    return appendNew<Value>(proc, opcode, origin, value);
}

Value* BasicBlock::appendNewControlValue(Procedure& proc, Opcode opcode, Origin origin, const FrequentedBlock& target)
{
    RELEASE_ASSERT(opcode == Jump);
    setSuccessors(target);
    return appendNew<Value>(proc, opcode, origin);
}

Value* BasicBlock::appendNewControlValue(Procedure& proc, Opcode opcode, Origin origin, Value* predicate, const FrequentedBlock& taken, const FrequentedBlock& notTaken)
{
    RELEASE_ASSERT(opcode == Branch);
    setSuccessors(taken, notTaken);
    return appendNew<Value>(proc, opcode, origin, predicate);
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

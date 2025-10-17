/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 23, 2024.
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

#include "AirFrequentedBlock.h"
#include "AirInst.h"
#include "B3SuccessorCollection.h"
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace JSC { namespace B3 {

template<typename> class GenericBlockInsertionSet;

namespace Air {

class BlockInsertionSet;
class Code;
class InsertionSet;
class PhaseInsertionSet;

class BasicBlock {
    WTF_MAKE_NONCOPYABLE(BasicBlock);
    WTF_MAKE_TZONE_ALLOCATED(BasicBlock);
public:
    static const char* const dumpPrefix;
    static constexpr unsigned uninsertedIndex = UINT_MAX;

    typedef Vector<Inst> InstList;
    typedef Vector<BasicBlock*, 2> PredecessorList;
    typedef Vector<FrequentedBlock, 2> SuccessorList;

    unsigned index() const { return m_index; }

    // This method is exposed for phases that mess with the layout of basic blocks. Currently that means just
    // optimizeBlockOrder().
    void setIndex(unsigned index) { m_index = index; }
    
    unsigned size() const { return m_insts.size(); }
    InstList::iterator begin() { return m_insts.begin(); }
    InstList::iterator end() { return m_insts.end(); }
    InstList::const_iterator begin() const { return m_insts.begin(); }
    InstList::const_iterator end() const { return m_insts.end(); }

    const Inst& at(unsigned index) const { return m_insts[index]; }
    Inst& at(unsigned index) { return m_insts[index]; }

    Inst* get(unsigned index)
    {
        return index < size() ? &at(index) : nullptr;
    }

    const Inst& last() const { return m_insts.last(); }
    Inst& last() { return m_insts.last(); }

    void resize(unsigned size) { m_insts.resize(size); }

    const InstList& insts() const { return m_insts; }
    InstList& insts() { return m_insts; }

    template<typename Inst>
    Inst& appendInst(Inst&& inst)
    {
        m_insts.append(std::forward<Inst>(inst));
        return m_insts.last();
    }

    template<typename... Arguments>
    Inst& append(Arguments&&... arguments)
    {
        m_insts.append(Inst(std::forward<Arguments>(arguments)...));
        return m_insts.last();
    }

    // The "0" case is the case to which the branch jumps, so the "then" case. The "1" case is the
    // "else" case, and is used to represent the fall-through of a conditional branch.
    unsigned numSuccessors() const { return m_successors.size(); }
    FrequentedBlock successor(unsigned index) const { return m_successors[index]; }
    FrequentedBlock& successor(unsigned index) { return m_successors[index]; }
    const SuccessorList& successors() const { return m_successors; }
    SuccessorList& successors() { return m_successors; }

    JS_EXPORT_PRIVATE void setSuccessors(FrequentedBlock);
    JS_EXPORT_PRIVATE void setSuccessors(FrequentedBlock, FrequentedBlock);

    BasicBlock* successorBlock(unsigned index) const { return successor(index).block(); }
    BasicBlock*& successorBlock(unsigned index) { return successor(index).block(); }
    SuccessorCollection<BasicBlock, SuccessorList> successorBlocks()
    {
        return SuccessorCollection<BasicBlock, SuccessorList>(m_successors);
    }
    SuccessorCollection<const BasicBlock, const SuccessorList> successorBlocks() const
    {
        return SuccessorCollection<const BasicBlock, const SuccessorList>(m_successors);
    }

    unsigned numPredecessors() const { return m_predecessors.size(); }
    BasicBlock* predecessor(unsigned index) const { return m_predecessors[index]; }
    BasicBlock*& predecessor(unsigned index) { return m_predecessors[index]; }
    const PredecessorList& predecessors() const { return m_predecessors; }
    PredecessorList& predecessors() { return m_predecessors; }

    bool addPredecessor(BasicBlock*);
    bool removePredecessor(BasicBlock*);
    bool replacePredecessor(BasicBlock* from, BasicBlock* to);
    bool containsPredecessor(BasicBlock* predecessor) const { return m_predecessors.contains(predecessor); }

    double frequency() const { return m_frequency; }

    void dump(PrintStream&) const;
    void deepDump(PrintStream&) const;

    void dumpHeader(PrintStream&) const;
    void dumpFooter(PrintStream&) const;

private:
    friend class BlockInsertionSet;
    friend class Code;
    friend class InsertionSet;
    friend class PhaseInsertionSet;
    template<typename> friend class B3::GenericBlockInsertionSet;
    
    BasicBlock(unsigned index, double frequency);

    unsigned m_index;
    InstList m_insts;
    SuccessorList m_successors;
    PredecessorList m_predecessors;
    double m_frequency;
};

class DeepBasicBlockDump {
public:
    DeepBasicBlockDump(const BasicBlock* block)
        : m_block(block)
    {
    }

    void dump(PrintStream& out) const
    {
        if (m_block)
            m_block->deepDump(out);
        else
            out.print("<null>");
    }

private:
    const BasicBlock* m_block;
};

inline DeepBasicBlockDump deepDump(const BasicBlock* block)
{
    return DeepBasicBlockDump(block);
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)

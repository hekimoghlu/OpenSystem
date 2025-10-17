/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 6, 2023.
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

#include "StaticPropertyAnalysis.h"
#include <wtf/HashMap.h>

namespace JSC {

// Used for flow-insensitive static analysis of the number of properties assigned to an object.
// We use this analysis with other runtime data to produce an optimization guess. This analysis
// is understood to be lossy, and it's OK if it turns out to be wrong sometimes.
class StaticPropertyAnalyzer {
public:
    void createThis(RegisterID* dst, JSInstructionStream::MutableRef instructionRef);
    void newObject(RegisterID* dst, JSInstructionStream::MutableRef instructionRef);
    void putById(RegisterID* dst, unsigned propertyIndex); // propertyIndex is an index into a uniqued set of strings.
    void mov(RegisterID* dst, RegisterID* src);

    void kill();
    void kill(RegisterID* dst);

private:
    void kill(StaticPropertyAnalysis*);

    typedef UncheckedKeyHashMap<int, RefPtr<StaticPropertyAnalysis>, WTF::IntHash<int>, WTF::UnsignedWithZeroKeyHashTraits<int>> AnalysisMap;
    AnalysisMap m_analyses;
};

inline void StaticPropertyAnalyzer::createThis(RegisterID* dst, JSInstructionStream::MutableRef instructionRef)
{
    AnalysisMap::AddResult addResult = m_analyses.add(
        dst->index(), StaticPropertyAnalysis::create(WTFMove(instructionRef)));
    ASSERT_UNUSED(addResult, addResult.isNewEntry); // Can't have two 'this' in the same constructor.
}

inline void StaticPropertyAnalyzer::newObject(RegisterID* dst, JSInstructionStream::MutableRef instructionRef)
{
    auto analysis = StaticPropertyAnalysis::create(WTFMove(instructionRef));
    AnalysisMap::AddResult addResult = m_analyses.add(dst->index(), analysis.copyRef());
    if (!addResult.isNewEntry) {
        kill(addResult.iterator->value.get());
        addResult.iterator->value = WTFMove(analysis);
    }
}

inline void StaticPropertyAnalyzer::putById(RegisterID* dst, unsigned propertyIndex)
{
    StaticPropertyAnalysis* analysis = m_analyses.get(dst->index());
    if (!analysis)
        return;
    analysis->addPropertyIndex(propertyIndex);
}

inline void StaticPropertyAnalyzer::mov(RegisterID* dst, RegisterID* src)
{
    RefPtr<StaticPropertyAnalysis> analysis = m_analyses.get(src->index());
    if (!analysis) {
        kill(dst);
        return;
    }

    AnalysisMap::AddResult addResult = m_analyses.add(dst->index(), analysis);
    if (!addResult.isNewEntry) {
        kill(addResult.iterator->value.get());
        addResult.iterator->value = WTFMove(analysis);
    }
}

inline void StaticPropertyAnalyzer::kill(StaticPropertyAnalysis* analysis)
{
    if (!analysis)
        return;
    if (!analysis->hasOneRef()) // Aliases for this object still exist, so it might acquire more properties.
        return;
    analysis->record();
}

inline void StaticPropertyAnalyzer::kill(RegisterID* dst)
{
    // We observe kills in order to avoid piling on properties to an object after
    // its bytecode register has been recycled.

    // Consider these cases:

    // (1) Aliased temporary
    // var o1 = { name: name };
    // var o2 = { name: name };

    // (2) Aliased local -- no control flow
    // var local;
    // local = new Object;
    // local.name = name;
    // ...

    // local = lookup();
    // local.didLookup = true;
    // ...

    // (3) Aliased local -- control flow
    // var local;
    // if (condition)
    //     local = { };
    // else {
    //     local = new Object;
    // }
    // local.name = name;

    // (Note: our default codegen for "new Object" looks like case (3).)

    // Case (1) is easy because temporaries almost never survive across control flow.

    // Cases (2) and (3) are hard. Case (2) should kill "local", while case (3) should
    // not. There is no great way to solve these cases with simple static analysis.

    // Since this is a simple static analysis, we just try to catch the simplest cases,
    // so we accept kills to any registers except for registers that have no inferred
    // properties yet.

    AnalysisMap::iterator it = m_analyses.find(dst->index());
    if (it == m_analyses.end())
        return;
    if (!it->value->propertyIndexCount())
        return;

    kill(it->value.get());
    m_analyses.remove(it);
}

inline void StaticPropertyAnalyzer::kill()
{
    while (m_analyses.size())
        kill(m_analyses.take(m_analyses.begin()->key).get());
}

} // namespace JSC

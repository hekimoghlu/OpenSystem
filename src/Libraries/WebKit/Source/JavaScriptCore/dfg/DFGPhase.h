/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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

#if ENABLE(DFG_JIT)

#include "CompilerTimingScope.h"
#include "DFGCommon.h"
#include "DFGGraph.h"

namespace JSC { namespace DFG {

class Phase {
public:
    Phase(Graph& graph, ASCIILiteral name, const bool disableGraphValidation = false)
        : m_graph(graph)
        , m_name(name)
        , m_disableGraphValidation(disableGraphValidation)
    {
        beginPhase();
    }
    
    ~Phase()
    {
        endPhase();
    }
    
    ASCIILiteral name() const { return m_name; }
    
    Graph& graph() { return m_graph; }
    
    // Each phase must have a run() method.
    
    Prefix prefix;

protected:
    // Things you need to have a DFG compiler phase.
    Graph& m_graph;
    
    VM& vm() { return m_graph.m_vm; }
    CodeBlock* codeBlock() { return m_graph.m_codeBlock; }
    CodeBlock* profiledBlock() { return m_graph.m_profiledBlock; }

    // This runs validation, and uses the graph dump before the phase if possible.
    void validate();
    
    ASCIILiteral m_name;
    
private:
    // Call these hooks when starting and finishing.
    void beginPhase();
    void endPhase();

    bool m_disableGraphValidation { false };
    CString m_graphDumpBeforePhase;
};

template<typename PhaseType>
bool runAndLog(PhaseType& phase)
{
    CompilerTimingScope timingScope("DFG"_s, phase.name());
    
    bool result = phase.run();

    if (result && logCompilationChanges(phase.graph().m_plan.mode()))
        dataLogLn(phase.graph().prefix(), "Phase ", phase.name(), " changed the IR.\n");
    return result;
}

template<typename PhaseType, typename... Args>
bool runPhase(Graph& graph, Args... args)
{
    PhaseType phase(graph, args...);
    return runAndLog(phase);
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

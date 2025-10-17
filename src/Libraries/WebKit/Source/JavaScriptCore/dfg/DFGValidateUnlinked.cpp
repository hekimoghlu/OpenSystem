/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 21, 2023.
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
#include "DFGValidateUnlinked.h"

#if ENABLE(DFG_JIT)

#include "DFGGraph.h"
#include "DFGPhase.h"
#include "DFGPhiChildren.h"
#include "JSCJSValueInlines.h"

namespace JSC::DFG {

class ValidateUnlinked final : public Phase {
    static constexpr bool verbose = false;

public:
    ValidateUnlinked(Graph& graph)
        : Phase(graph, "uDFG validation"_s)
    {
    }

    bool run();
    bool validateNode(Node*);

private:
};

bool ValidateUnlinked::run()
{
    for (BasicBlock* block : m_graph.blocksInPreOrder()) {
        for (Node* node : *block) {
            if (!validateNode(node))
                return false;
        }
    }
    return true;
}

bool ValidateUnlinked::validateNode(Node* node)
{
    JSGlobalObject* globalObject = m_graph.globalObjectFor(node->origin.semantic);
    if (globalObject != m_graph.m_codeBlock->globalObject()) {
        if (UNLIKELY(Options::dumpUnlinkedDFGValidation())) {
            m_graph.logAssertionFailure(node, __FILE__, __LINE__, WTF_PRETTY_FUNCTION, "Bad GlobalObject");
            dataLogLn(RawPointer(globalObject), " != ", RawPointer(m_graph.m_codeBlock->globalObject()));
        }
        return false;
    }
    return true;
}


CapabilityLevel canCompileUnlinked(Graph& graph)
{
    ValidateUnlinked phase(graph);
    if (!phase.run())
        return CapabilityLevel::CannotCompile;
    return CapabilityLevel::CanCompileAndInline;
}

} // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 9, 2023.
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
#include "FTLState.h"

#if ENABLE(FTL_JIT)

#include "AirCode.h"
#include "AirDisassembler.h"
#include "B3ValueInlines.h"
#include "CodeBlockWithJITType.h"
#include "FTLForOSREntryJITCode.h"
#include "FTLJITCode.h"
#include "FTLJITFinalizer.h"
#include "FTLPatchpointExceptionHandle.h"

#include <wtf/RecursableLambda.h>

namespace JSC { namespace FTL {

using namespace B3;
using namespace DFG;

State::State(Graph& graph)
    : graph(graph)
{
    switch (graph.m_plan.mode()) {
    case JITCompilationMode::FTL: {
        jitCode = adoptRef(new JITCode());
        break;
    }
    case JITCompilationMode::FTLForOSREntry: {
        RefPtr<ForOSREntryJITCode> code = adoptRef(new ForOSREntryJITCode());
        code->initializeEntryBuffer(graph.m_vm, graph.m_profiledBlock->numCalleeLocals());
        code->setBytecodeIndex(graph.m_plan.osrEntryBytecodeIndex());
        jitCode = code;
        break;
    }
    default:
        RELEASE_ASSERT_NOT_REACHED();
        break;
    }

    graph.m_plan.setFinalizer(makeUnique<JITFinalizer>(graph.m_plan));
    finalizer = static_cast<JITFinalizer*>(graph.m_plan.finalizer());

    proc = makeUnique<Procedure>(/* usesSIMD = */ false);

    if (graph.m_vm.shouldBuilderPCToCodeOriginMapping())
        proc->setNeedsPCToOriginMap();

    proc->setOriginPrinter(
        [] (PrintStream& out, B3::Origin origin) {
            out.print(std::bit_cast<Node*>(origin.data()));
        });

    proc->setFrontendData(&graph);
}

void State::dumpDisassembly(PrintStream& out, LinkBuffer& linkBuffer, const ScopedLambda<void(DFG::Node*)>& perDFGNodeCallback)
{
    B3::Air::Disassembler* disassembler = proc->code().disassembler();

    out.print("Generated ", graph.m_plan.mode(), " code for ", CodeBlockWithJITType(graph.m_codeBlock, JITType::FTLJIT), ", instructions size = ", graph.m_codeBlock->instructionsSize(), ":\n");

    B3::Value* currentB3Value = nullptr;
    Node* currentDFGNode = nullptr;

    UncheckedKeyHashSet<B3::Value*> printedValues;
    UncheckedKeyHashSet<Node*> printedNodes;
    const char* dfgPrefix = "DFG " "    ";
    const char* b3Prefix  = "b3  " "          ";
    const char* airPrefix = "Air " "              ";
    const char* asmPrefix = "asm " "                ";

    auto printDFGNode = [&] (Node* node) {
        if (currentDFGNode == node)
            return;

        currentDFGNode = node;
        if (!currentDFGNode)
            return;

        perDFGNodeCallback(node);

        UncheckedKeyHashSet<Node*> localPrintedNodes;
        WTF::Function<void(Node*)> printNodeRecursive = [&] (Node* node) {
            if (printedNodes.contains(node) || localPrintedNodes.contains(node))
                return;

            localPrintedNodes.add(node);
            graph.doToChildren(node, [&] (Edge child) {
                printNodeRecursive(child.node());
            });
            graph.dump(out, dfgPrefix, node);
        };
        printNodeRecursive(node);
        printedNodes.add(node);
    };

    auto printB3Value = [&] (B3::Value* value) {
        if (currentB3Value == value)
            return;

        currentB3Value = value;
        if (!currentB3Value)
            return;

        printDFGNode(std::bit_cast<Node*>(value->origin().data()));

        UncheckedKeyHashSet<B3::Value*> localPrintedValues;
        auto printValueRecursive = recursableLambda([&] (auto self, B3::Value* value) -> void {
            if (printedValues.contains(value) || localPrintedValues.contains(value))
                return;

            localPrintedValues.add(value);
            for (unsigned i = 0; i < value->numChildren(); i++)
                self(value->child(i));
            out.print(b3Prefix);
            value->deepDump(proc.get(), out);
            out.print("\n");
        });

        printValueRecursive(currentB3Value);
        printedValues.add(value);
    };

    B3::Value* prevOrigin = nullptr;
    auto forEachInst = scopedLambda<void(B3::Air::Inst&)>([&] (B3::Air::Inst& inst) {
        if (inst.origin != prevOrigin) {
            printB3Value(inst.origin);
            prevOrigin = inst.origin;
        }
    });

    disassembler->dump(proc->code(), out, linkBuffer, airPrefix, asmPrefix, forEachInst);
    linkBuffer.didAlreadyDisassemble();
}

State::~State() = default;

StructureStubInfo* State::addStructureStubInfo()
{
    ASSERT(!graph.m_plan.isUnlinked());
    auto* stubInfo = jitCode->common.m_stubInfos.add();
    stubInfo->useDataIC = Options::useDataICInFTL();
    return stubInfo;
}

OptimizingCallLinkInfo* State::addCallLinkInfo(CodeOrigin codeOrigin)
{
    return jitCode->common.m_callLinkInfos.add(codeOrigin, graph.m_codeBlock);
}

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)


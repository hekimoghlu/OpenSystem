/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 13, 2023.
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
#include "FTLCompile.h"

#if ENABLE(FTL_JIT)

#include "AirCode.h"
#include "AirDisassembler.h"
#include "AirStackSlot.h"
#include "B3Generate.h"
#include "B3Value.h"
#include "B3ValueInlines.h"
#include "CodeBlockWithJITType.h"
#include "CCallHelpers.h"
#include "DFGGraphSafepoint.h"
#include "FTLJITCode.h"
#include "JITThunks.h"
#include "LLIntEntrypoint.h"
#include "LLIntThunks.h"
#include "LinkBuffer.h"
#include "PCToCodeOriginMap.h"
#include "ThunkGenerators.h"
#include <wtf/RecursableLambda.h>
#include <wtf/SetForScope.h>

namespace JSC { namespace FTL {

const char* const tierName = "FTL ";

using namespace DFG;

void compile(State& state, Safepoint::Result& safepointResult)
{
    Graph& graph = state.graph;
    CodeBlock* codeBlock = graph.m_codeBlock;
    VM& vm = graph.m_vm;

    if (shouldDumpDisassembly() || vm.m_perBytecodeProfiler)
        state.proc->code().setDisassembler(makeUnique<B3::Air::Disassembler>());

    if (!shouldDumpDisassembly() && !verboseCompilationEnabled() && !Options::verboseValidationFailure() && !Options::asyncDisassembly() && !graph.compilation() && !state.proc->needsPCToOriginMap())
        graph.freeDFGIRAfterLowering();

    {
        GraphSafepoint safepoint(state.graph, safepointResult);
        B3::prepareForGeneration(*state.proc);
    }
    if (safepointResult.didGetCancelled())
        return;
    RELEASE_ASSERT(!state.graph.m_vm.heap.worldIsStopped());
    
    if (state.allocationFailed)
        return;
    
    RegisterAtOffsetList registerOffsets = state.proc->calleeSaveRegisterAtOffsetList();
    if (shouldDumpDisassembly())
        dataLog(tierName, "Unwind info for ", CodeBlockWithJITType(codeBlock, JITType::FTLJIT), ": ", registerOffsets, "\n");
    state.jitCode->m_calleeSaveRegisters = RegisterAtOffsetList(WTFMove(registerOffsets));
    ASSERT(!(state.proc->frameSize() % sizeof(EncodedJSValue)));
    state.jitCode->common.frameRegisterCount = state.proc->frameSize() / sizeof(EncodedJSValue);

    int localsOffset =
        state.capturedValue->offsetFromFP() / sizeof(EncodedJSValue) + graph.m_nextMachineLocal;
    if (shouldDumpDisassembly()) {
        dataLog(tierName,
            "localsOffset = ", localsOffset, " for stack slot: ",
            pointerDump(state.capturedValue), " at ", RawPointer(state.capturedValue), "\n");
    }
    
    for (unsigned i = graph.m_inlineVariableData.size(); i--;) {
        InlineCallFrame* inlineCallFrame = graph.m_inlineVariableData[i].inlineCallFrame;
        
        if (inlineCallFrame->argumentCountRegister.isValid())
            inlineCallFrame->argumentCountRegister += localsOffset;
        
        for (unsigned argument = inlineCallFrame->m_argumentsWithFixup.size(); argument-- > 1;) {
            inlineCallFrame->m_argumentsWithFixup[argument] =
                inlineCallFrame->m_argumentsWithFixup[argument].withLocalsOffset(localsOffset);
        }
        
        if (inlineCallFrame->isClosureCall) {
            inlineCallFrame->calleeRecovery =
                inlineCallFrame->calleeRecovery.withLocalsOffset(localsOffset);
        }

    }

    // Note that the scope register could be invalid here if the original code had CallDirectEval but it
    // got killed. That's because it takes the CallDirectEval to cause the scope register to be kept alive
    // unless the debugger is also enabled.
    if (graph.needsScopeRegister() && codeBlock->scopeRegister().isValid())
        codeBlock->setScopeRegister(codeBlock->scopeRegister() + localsOffset);

    for (OSRExitDescriptor& descriptor : state.jitCode->osrExitDescriptors) {
        for (unsigned i = descriptor.m_values.size(); i--;)
            descriptor.m_values[i] = descriptor.m_values[i].withLocalsOffset(localsOffset);
        for (ExitTimeObjectMaterialization* materialization : descriptor.m_materializations)
            materialization->accountForLocalsOffset(localsOffset);
    }

    // We will add exception handlers while generating.
    codeBlock->clearExceptionHandlers();

    CCallHelpers jit(codeBlock);
    {
        GraphSafepoint safepoint(state.graph, safepointResult, true);
        B3::generate(*state.proc, jit);
    }
    if (safepointResult.didGetCancelled())
        return;

    // Emit the exception handler.
    *state.exceptionHandler = jit.label();
    jit.jumpThunk(CodeLocationLabel(vm.getCTIStub(CommonJITThunkID::HandleException).template retaggedCode<NoPtrTag>()));

    CCallHelpers::Label mainPathLabel = state.proc->code().entrypointLabel(0);
    CCallHelpers::Label entryLabel = mainPathLabel;
    CCallHelpers::Label arityCheckLabel = mainPathLabel;

    // Generating entrypoints.
    switch (state.graph.m_plan.mode()) {
    case JITCompilationMode::FTL: {
        bool requiresArityFixup = codeBlock->numParameters() != 1;
        if (codeBlock->codeType() == FunctionCode && requiresArityFixup) {
            CCallHelpers::JumpList mainPathJumps;

            arityCheckLabel = jit.label();
            jit.load32(CCallHelpers::calleeFramePayloadSlot(CallFrameSlot::argumentCountIncludingThis).withOffset(sizeof(CallerFrameAndPC) - prologueStackPointerDelta()), GPRInfo::argumentGPR2);
            mainPathJumps.append(jit.branch32(CCallHelpers::AboveOrEqual, GPRInfo::argumentGPR2, CCallHelpers::TrustedImm32(codeBlock->numParameters())));

            unsigned numberOfParameters = codeBlock->numParameters();
            CCallHelpers::JumpList stackOverflowWithEntry;
            jit.getArityPadding(vm, numberOfParameters, GPRInfo::argumentGPR2, GPRInfo::argumentGPR0, GPRInfo::argumentGPR1, GPRInfo::argumentGPR3, stackOverflowWithEntry);

#if CPU(X86_64)
            jit.pop(GPRInfo::argumentGPR1);
#else
            jit.tagPtr(NoPtrTag, CCallHelpers::linkRegister);
            jit.move(CCallHelpers::linkRegister, GPRInfo::argumentGPR1);
#endif
            jit.nearCallThunk(CodeLocationLabel { LLInt::arityFixup() });
#if CPU(X86_64)
            jit.push(GPRInfo::argumentGPR1);
#else
            jit.move(GPRInfo::argumentGPR1, CCallHelpers::linkRegister);
            jit.untagPtr(NoPtrTag, CCallHelpers::linkRegister);
            jit.validateUntaggedPtr(CCallHelpers::linkRegister, GPRInfo::argumentGPR0);
#endif
            mainPathJumps.append(jit.jump());

            stackOverflowWithEntry.link(&jit);
            jit.emitFunctionPrologue();
            jit.move(CCallHelpers::TrustedImmPtr(codeBlock), GPRInfo::argumentGPR0);
            jit.storePtr(GPRInfo::callFrameRegister, &vm.topCallFrame);
            jit.callOperation<OperationPtrTag>(operationThrowStackOverflowError);
            jit.jumpThunk(CodeLocationLabel(vm.getCTIStub(CommonJITThunkID::HandleExceptionWithCallFrameRollback).retaggedCode<NoPtrTag>()));
            mainPathJumps.linkTo(mainPathLabel, &jit);
        }
        break;
    }

    case JITCompilationMode::FTLForOSREntry: {
        // We jump to here straight from DFG code, after having boxed up all of the
        // values into the scratch buffer. Everything should be good to go - at this
        // point we've even done the stack check. Basically we just have to make the
        // call to the B3-generated code.
        entryLabel = jit.label();
        arityCheckLabel = entryLabel;
        jit.emitFunctionEpilogue();
        jit.untagReturnAddress();
        jit.jump().linkTo(mainPathLabel, &jit);
        break;
    }

    default:
        RELEASE_ASSERT_NOT_REACHED();
        break;
    }

    state.b3CodeLinkBuffer = makeUnique<LinkBuffer>(jit, codeBlock, LinkBuffer::Profile::FTL, JITCompilationCanFail);

    if (state.b3CodeLinkBuffer->didFailToAllocate()) {
        state.allocationFailed = true;
        return;
    }

    if (vm.shouldBuilderPCToCodeOriginMapping()) {
        B3::PCToOriginMap originMap = state.proc->releasePCToOriginMap();
        state.jitCode->common.m_pcToCodeOriginMap = makeUnique<PCToCodeOriginMap>(PCToCodeOriginMapBuilder(PCToCodeOriginMapBuilder::JSCodeOriginMap, vm, WTFMove(originMap)), *state.b3CodeLinkBuffer);
    }

    state.jitCode->initializeAddressForCall(state.b3CodeLinkBuffer->locationOf<JSEntryPtrTag>(entryLabel));
    state.jitCode->initializeAddressForArityCheck(state.b3CodeLinkBuffer->locationOf<JSEntryPtrTag>(arityCheckLabel));
    state.jitCode->initializeB3Byproducts(state.proc->releaseByproducts());

    for (auto pair : state.graph.m_entrypointIndexToCatchBytecodeIndex) {
        BytecodeIndex catchBytecodeIndex = pair.value;
        unsigned entrypointIndex = pair.key;
        Vector<FlushFormat> argumentFormats = state.graph.m_argumentFormats[entrypointIndex];
        state.graph.appendCatchEntrypoint(catchBytecodeIndex, state.b3CodeLinkBuffer->locationOf<ExceptionHandlerPtrTag>(state.proc->code().entrypointLabel(entrypointIndex)), WTFMove(argumentFormats));
    }
    state.jitCode->common.finalizeCatchEntrypoints(WTFMove(state.graph.m_catchEntrypoints));

    if (shouldDumpDisassembly())
        state.dumpDisassembly(WTF::dataFile(), *state.b3CodeLinkBuffer);

    Profiler::Compilation* compilation = graph.compilation();
    if (UNLIKELY(compilation)) {
        compilation->addDescription(
            Profiler::OriginStack(),
            toCString("Generated FTL DFG IR for ", CodeBlockWithJITType(codeBlock, JITType::FTLJIT), ", instructions size = ", graph.m_codeBlock->instructionsSize(), ":\n"));

        graph.ensureSSADominators();
        graph.ensureSSANaturalLoops();

        constexpr auto prefix = "    "_s;

        DumpContext dumpContext;
        StringPrintStream out;
        Node* lastNode = nullptr;
        for (size_t blockIndex = 0; blockIndex < graph.numBlocks(); ++blockIndex) {
            DFG::BasicBlock* block = graph.block(blockIndex);
            if (!block)
                continue;

            graph.dumpBlockHeader(out, prefix, block, Graph::DumpLivePhisOnly, &dumpContext);
            compilation->addDescription(Profiler::OriginStack(), out.toCString());
            out.reset();

            for (size_t nodeIndex = 0; nodeIndex < block->size(); ++nodeIndex) {
                Node* node = block->at(nodeIndex);

                Profiler::OriginStack stack;

                if (node->origin.semantic.isSet()) {
                    stack = Profiler::OriginStack(
                        *vm.m_perBytecodeProfiler, codeBlock, node->origin.semantic);
                }

                if (graph.dumpCodeOrigin(out, prefix, lastNode, node, &dumpContext)) {
                    compilation->addDescription(stack, out.toCString());
                    out.reset();
                }

                graph.dump(out, prefix, node, &dumpContext);
                compilation->addDescription(stack, out.toCString());
                out.reset();

                if (node->origin.semantic.isSet())
                    lastNode = node;
            }
        }

        dumpContext.dump(out, prefix);
        compilation->addDescription(Profiler::OriginStack(), out.toCString());
        out.reset();

        out.print("\n\n\n    FTL B3/Air Disassembly:\n");
        compilation->addDescription(Profiler::OriginStack(), out.toCString());
        out.reset();

        state.dumpDisassembly(out, *state.b3CodeLinkBuffer, scopedLambda<void(Node*)>([&] (Node*) {
            compilation->addDescription({ }, out.toCString());
            out.reset();
        }));
        compilation->addDescription({ }, out.toCString());
        out.reset();

        state.jitCode->common.compilation = compilation;
    }

}

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)


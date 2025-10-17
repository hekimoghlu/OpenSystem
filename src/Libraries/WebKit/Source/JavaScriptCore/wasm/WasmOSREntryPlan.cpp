/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 11, 2021.
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
#include "WasmOSREntryPlan.h"

#if ENABLE(WEBASSEMBLY_OMGJIT)

#include "JITCompilation.h"
#include "LinkBuffer.h"
#include "NativeCalleeRegistry.h"
#include "WasmCallee.h"
#include "WasmIRGeneratorHelpers.h"
#include "WasmMachineThreads.h"
#include "WasmNameSection.h"
#include "WasmOMGIRGenerator.h"
#include "WasmTypeDefinitionInlines.h"
#include <wtf/DataLog.h>
#include <wtf/Locker.h>
#include <wtf/StdLibExtras.h>
#include <wtf/text/MakeString.h>

namespace JSC { namespace Wasm {

namespace WasmOSREntryPlanInternal {
static constexpr bool verbose = false;
}

OSREntryPlan::OSREntryPlan(VM& vm, Ref<Module>&& module, Ref<Callee>&& callee, FunctionCodeIndex functionIndex, std::optional<bool> hasExceptionHandlers, uint32_t loopIndex, MemoryMode mode, CompletionTask&& task)
    : Base(vm, const_cast<ModuleInformation&>(module->moduleInformation()), WTFMove(task))
    , m_module(WTFMove(module))
    , m_calleeGroup(*m_module->calleeGroupFor(mode))
    , m_callee(WTFMove(callee))
    , m_hasExceptionHandlers(hasExceptionHandlers)
    , m_functionIndex(functionIndex)
    , m_loopIndex(loopIndex)
{
    ASSERT(Options::useOMGJIT());
    setMode(mode);
    ASSERT(m_calleeGroup->runnable());
    ASSERT(m_calleeGroup.ptr() == m_module->calleeGroupFor(m_mode));
    dataLogLnIf(WasmOSREntryPlanInternal::verbose, "Starting OMGForOSREntry plan for ", functionIndex, " of module: ", RawPointer(&m_module.get()));
}

void OSREntryPlan::dumpDisassembly(CompilationContext& context, LinkBuffer& linkBuffer, FunctionCodeIndex functionIndex, const TypeDefinition& signature, FunctionSpaceIndex functionIndexSpace)
{
    CompilationMode targetCompilationMode = CompilationMode::OMGForOSREntryMode;
    dataLogLnIf(context.procedure->shouldDumpIR() || shouldDumpDisassemblyFor(targetCompilationMode), "Generated OMG code for WebAssembly OMGforOSREntry function[", functionIndex, "] ", signature.toString().ascii().data(), " name ", makeString(IndexOrName(functionIndexSpace, m_moduleInformation->nameSection->get(functionIndexSpace))).ascii().data());
    if (UNLIKELY(shouldDumpDisassemblyFor(targetCompilationMode))) {
        auto* disassembler = context.procedure->code().disassembler();

        const char* b3Prefix = "b3    ";
        const char* airPrefix = "Air        ";
        const char* asmPrefix = "asm              ";

        B3::Value* prevOrigin = nullptr;
        auto forEachInst = scopedLambda<void(B3::Air::Inst&)>([&] (B3::Air::Inst& inst) {
            if (inst.origin && inst.origin != prevOrigin && context.procedure->code().shouldPreserveB3Origins()) {
                if (String string = inst.origin->compilerConstructionSite(); !string.isNull())
                    dataLogLn(string);
                dataLog(b3Prefix);
                inst.origin->deepDump(context.procedure.get(), WTF::dataFile());
                dataLogLn();
                prevOrigin = inst.origin;
            }
        });

        disassembler->dump(context.procedure->code(), WTF::dataFile(), linkBuffer, airPrefix, asmPrefix, forEachInst);
        linkBuffer.didAlreadyDisassemble();
    }
}

void OSREntryPlan::work(CompilationEffort)
{
    ASSERT(m_calleeGroup->runnable());
    ASSERT(m_calleeGroup.ptr() == m_module->calleeGroupFor(mode()));
    const FunctionData& function = m_moduleInformation->functions[m_functionIndex];

    const FunctionSpaceIndex functionIndexSpace = m_module->moduleInformation().toSpaceIndex(m_functionIndex);
    ASSERT(functionIndexSpace < m_module->moduleInformation().functionIndexSpaceSize());

    TypeIndex typeIndex = m_moduleInformation->internalFunctionTypeIndices[m_functionIndex];
    const TypeDefinition& signature = TypeInformation::get(typeIndex).expand();

    Ref<OMGOSREntryCallee> callee = OMGOSREntryCallee::create(functionIndexSpace, m_moduleInformation->nameSection->get(functionIndexSpace), m_loopIndex);

    beginCompilerSignpost(callee.get());
    Vector<UnlinkedWasmToWasmCall> unlinkedCalls;
    CompilationContext context;
    auto parseAndCompileResult = parseAndCompileOMG(context, callee.get(), function, signature, unlinkedCalls, m_calleeGroup.get(), m_moduleInformation.get(), m_mode, CompilationMode::OMGForOSREntryMode, m_functionIndex, m_hasExceptionHandlers, m_loopIndex);
    endCompilerSignpost(callee.get());

    if (UNLIKELY(!parseAndCompileResult)) {
        Locker locker { m_lock };
        fail(makeString(parseAndCompileResult.error(), "when trying to tier up "_s, m_functionIndex.rawIndex()));
        return;
    }

    Entrypoint omgEntrypoint;
    LinkBuffer linkBuffer(*context.wasmEntrypointJIT, callee.ptr(), LinkBuffer::Profile::WasmOMG, JITCompilationCanFail);
    if (UNLIKELY(linkBuffer.didFailToAllocate())) {
        Locker locker { m_lock };
        Base::fail(makeString("Out of executable memory while tiering up function at index "_s, m_functionIndex.rawIndex()));
        return;
    }

    InternalFunction* internalFunction = parseAndCompileResult->get();
    Vector<CodeLocationLabel<ExceptionHandlerPtrTag>> exceptionHandlerLocations;
    computeExceptionHandlerLocations(exceptionHandlerLocations, internalFunction, context, linkBuffer);

    auto samplingProfilerMap = callee->materializePCToOriginMap(context.procedure->releasePCToOriginMap(), linkBuffer);

    dumpDisassembly(context, linkBuffer, m_functionIndex, signature, functionIndexSpace);
    omgEntrypoint.compilation = makeUnique<Compilation>(
        FINALIZE_CODE_IF(context.procedure->shouldDumpIR(), linkBuffer, JITCompilationPtrTag, nullptr, "WebAssembly OMGForOSREntry function[%i] %s name %s", m_functionIndex, signature.toString().ascii().data(), makeString(IndexOrName(functionIndexSpace, m_moduleInformation->nameSection->get(functionIndexSpace))).ascii().data()),
        WTFMove(context.wasmEntrypointByproducts));

    omgEntrypoint.calleeSaveRegisters = WTFMove(internalFunction->entrypoint.calleeSaveRegisters);

    ASSERT(m_calleeGroup.ptr() == m_module->calleeGroupFor(mode()));
    callee->setEntrypoint(WTFMove(omgEntrypoint), internalFunction->osrEntryScratchBufferSize, WTFMove(unlinkedCalls), WTFMove(internalFunction->stackmaps), WTFMove(internalFunction->exceptionHandlers), WTFMove(exceptionHandlerLocations));

    if (samplingProfilerMap)
        NativeCalleeRegistry::singleton().addPCToCodeOriginMap(callee.ptr(), WTFMove(samplingProfilerMap));

    {
        Locker locker { m_calleeGroup->m_lock };
        m_calleeGroup->recordOMGOSREntryCallee(locker, m_functionIndex, callee.get());
        m_calleeGroup->reportCallees(locker, callee.ptr(), internalFunction->outgoingJITDirectCallees);

        for (auto& call : callee->wasmToWasmCallsites()) {
            CodePtr<WasmEntryPtrTag> entrypoint;
            RefPtr<Wasm::Callee> wasmCallee;
            if (call.functionIndexSpace < m_module->moduleInformation().importFunctionCount())
                entrypoint = m_calleeGroup->m_wasmToWasmExitStubs[call.functionIndexSpace].code();
            else {
                wasmCallee = m_calleeGroup->wasmEntrypointCalleeFromFunctionIndexSpace(locker, call.functionIndexSpace);
                entrypoint = wasmCallee->entrypoint().retagged<WasmEntryPtrTag>();
            }

            MacroAssembler::repatchNearCall(call.callLocation, CodeLocationLabel<WasmEntryPtrTag>(entrypoint));
            MacroAssembler::repatchPointer(call.calleeLocation, CalleeBits::boxNativeCalleeIfExists(wasmCallee.get()));
        }

        resetInstructionCacheOnAllThreads();
        WTF::storeStoreFence();

        {
            switch (m_callee->compilationMode()) {
            case CompilationMode::BBQMode: {
                BBQCallee* bbqCallee = static_cast<BBQCallee*>(m_callee.ptr());
                Locker locker { bbqCallee->tierUpCounter().getLock() };
                bbqCallee->setOSREntryCallee(callee.copyRef(), mode());
                bbqCallee->tierUpCounter().osrEntryTriggers()[m_loopIndex] = TierUpCount::TriggerReason::CompilationDone;
                bbqCallee->tierUpCounter().setCompilationStatusForOMGForOSREntry(mode(), TierUpCount::CompilationStatus::Compiled);
                break;
            }
            default:
                RELEASE_ASSERT_NOT_REACHED();
            }
        }
    }

    // We don't register our BBQCallee for deletion because this entrypoint isn't a general one and is only used for loop OSR.

    dataLogLnIf(WasmOSREntryPlanInternal::verbose, "Finished OMGForOSREntry ", m_functionIndex);
    Locker locker { m_lock };
    complete();
}

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY_OMGJIT)

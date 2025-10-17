/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 14, 2022.
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
#include "DFGDesiredWatchpoints.h"

#if ENABLE(DFG_JIT)

#include "CodeBlock.h"
#include "DFGGraph.h"
#include "JSCInlines.h"

namespace JSC { namespace DFG {

bool ArrayBufferViewWatchpointAdaptor::add(CodeBlock* codeBlock, JSArrayBufferView* view, WatchpointCollector& collector)
{
    return collector.addWatchpoint([&](CodeBlockJettisoningWatchpoint& watchpoint) {
        if (hasBeenInvalidated(view, collector.concurrency()))
            return false;

        // view is already frozen. If it is deallocated, jettisoning happens.
        watchpoint.initialize(codeBlock);
        ArrayBuffer* arrayBuffer = view->possiblySharedBuffer();
        if (!arrayBuffer) {
            watchpoint.fire(codeBlock->vm(), StringFireDetail("ArrayBuffer could not be allocated, probably because of OOM."));
            return true;
        }

        // FIXME: We don't need to set this watchpoint at all for shared buffers.
        // https://bugs.webkit.org/show_bug.cgi?id=164108
        arrayBuffer->detachingWatchpointSet().add(&watchpoint);
        return true;
    });
}

bool SymbolTableAdaptor::add(CodeBlock* codeBlock, SymbolTable* symbolTable, WatchpointCollector& collector)
{
    return collector.addWatchpoint([&](CodeBlockJettisoningWatchpoint& watchpoint) {
        if (hasBeenInvalidated(symbolTable, collector.concurrency()))
            return false;

        // symbolTable is already frozen strongly.
        watchpoint.initialize(codeBlock);
        symbolTable->singleton().add(&watchpoint);
        return true;
    });
}

bool FunctionExecutableAdaptor::add(CodeBlock* codeBlock, FunctionExecutable* executable, WatchpointCollector& collector)
{
    return collector.addWatchpoint([&](CodeBlockJettisoningWatchpoint& watchpoint) {
        if (hasBeenInvalidated(executable, collector.concurrency()))
            return false;

        // executable is already frozen strongly.
        watchpoint.initialize(codeBlock);
        executable->singleton().add(&watchpoint);

        return true;
    });
}

bool AdaptiveStructureWatchpointAdaptor::add(CodeBlock* codeBlock, const ObjectPropertyCondition& key, WatchpointCollector& collector)
{
    VM& vm = codeBlock->vm();
    switch (key.kind()) {
    case PropertyCondition::Equivalence: {
        return collector.addAdaptiveInferredPropertyValueWatchpoint([&](AdaptiveInferredPropertyValueWatchpoint& watchpoint) {
            if (hasBeenInvalidated(key, collector.concurrency()))
                return false;

            watchpoint.initialize(key, codeBlock);
            watchpoint.install(vm);
            return true;
        });
    }
    default: {
        return collector.addAdaptiveStructureWatchpoint([&](AdaptiveStructureWatchpoint& watchpoint) {
            if (hasBeenInvalidated(key, collector.concurrency()))
                return false;

            watchpoint.initialize(key, codeBlock);
            watchpoint.install(vm);
            return true;
        });
    }
    }
    return true;
}

DesiredWatchpoints::DesiredWatchpoints() = default;
DesiredWatchpoints::~DesiredWatchpoints() = default;

void DesiredWatchpoints::addLazily(WatchpointSet& set)
{
    m_sets.addLazily(&set);
}

void DesiredWatchpoints::addLazily(InlineWatchpointSet& set)
{
    m_inlineSets.addLazily(&set);
}

void DesiredWatchpoints::addLazily(Graph& graph, SymbolTable* symbolTable)
{
    graph.freezeStrong(symbolTable); // Keep this strongly.
    m_symbolTables.addLazily(symbolTable);
}

void DesiredWatchpoints::addLazily(Graph& graph, FunctionExecutable* executable)
{
    graph.freezeStrong(executable); // Keep this strongly.
    m_functionExecutables.addLazily(executable);
}

void DesiredWatchpoints::addLazily(JSArrayBufferView* view)
{
    m_bufferViews.addLazily(view);
}

void DesiredWatchpoints::addLazily(const ObjectPropertyCondition& key)
{
    m_adaptiveStructureSets.addLazily(key);
}

void DesiredWatchpoints::addLazily(DesiredGlobalProperty&& property)
{
    m_globalProperties.addLazily(WTFMove(property));
}

bool DesiredWatchpoints::consider(Structure* structure)
{
    if (!structure->dfgShouldWatch())
        return false;
    addLazily(structure->transitionWatchpointSet());
    return true;
}

void DesiredWatchpoints::countWatchpoints(CodeBlock* codeBlock, DesiredIdentifiers& identifiers, CommonData* commonData)
{
    WatchpointCollector collector(*commonData);

    m_sets.reallyAdd(codeBlock, collector);
    m_inlineSets.reallyAdd(codeBlock, collector);
    m_symbolTables.reallyAdd(codeBlock, collector);
    m_functionExecutables.reallyAdd(codeBlock, collector);
    m_bufferViews.reallyAdd(codeBlock, collector);
    m_adaptiveStructureSets.reallyAdd(codeBlock, collector);
    m_globalProperties.reallyAdd(codeBlock, identifiers, collector);

    auto counts = collector.counts();
    commonData->m_watchpoints = FixedVector<CodeBlockJettisoningWatchpoint>(counts.m_watchpointCount);
    commonData->m_adaptiveStructureWatchpoints = FixedVector<AdaptiveStructureWatchpoint>(counts.m_adaptiveStructureWatchpointCount);
    commonData->m_adaptiveInferredPropertyValueWatchpoints = FixedVector<AdaptiveInferredPropertyValueWatchpoint>(counts.m_adaptiveInferredPropertyValueWatchpointCount);
}

bool DesiredWatchpoints::reallyAdd(CodeBlock* codeBlock, DesiredIdentifiers& identifiers, CommonData* commonData)
{
    WatchpointCollector collector(*commonData);
    collector.materialize();

    if (!m_sets.reallyAdd(codeBlock, collector))
        return false;

    if (!m_inlineSets.reallyAdd(codeBlock, collector))
        return false;

    if (!m_symbolTables.reallyAdd(codeBlock, collector))
        return false;

    if (!m_functionExecutables.reallyAdd(codeBlock, collector))
        return false;

    if (!m_bufferViews.reallyAdd(codeBlock, collector))
        return false;

    if (!m_adaptiveStructureSets.reallyAdd(codeBlock, collector))
        return false;

    if (!m_globalProperties.reallyAdd(codeBlock, identifiers, collector))
        return false;

    collector.finalize(*commonData);

    return true;
}

bool DesiredWatchpoints::areStillValidOnMainThread(VM& vm, DesiredIdentifiers& identifiers)
{
    return m_globalProperties.isStillValidOnMainThread(vm, identifiers);
}

void DesiredWatchpoints::dumpInContext(PrintStream& out, DumpContext* context) const
{
    Prefix noPrefix(Prefix::NoHeader);
    Prefix& prefix = context && context->graph ? context->graph->prefix() : noPrefix;
    out.print(prefix, "Desired watchpoints:\n");
    out.print(prefix, "    Watchpoint sets: ", inContext(m_sets, context), "\n");
    out.print(prefix, "    Inline watchpoint sets: ", inContext(m_inlineSets, context), "\n");
    out.print(prefix, "    SymbolTables: ", inContext(m_symbolTables, context), "\n");
    out.print(prefix, "    FunctionExecutables: ", inContext(m_functionExecutables, context), "\n");
    out.print(prefix, "    Buffer views: ", inContext(m_bufferViews, context), "\n");
    out.print(prefix, "    Object property conditions: ", inContext(m_adaptiveStructureSets, context), "\n");
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)


/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 27, 2022.
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
#include "InspectorHeapAgent.h"

#include "HeapProfiler.h"
#include "HeapSnapshot.h"
#include "InjectedScript.h"
#include "InjectedScriptManager.h"
#include "InspectorEnvironment.h"
#include "JSBigInt.h"
#include "VM.h"
#include <wtf/Stopwatch.h>
#include <wtf/TZoneMallocInlines.h>

namespace Inspector {

using namespace JSC;

WTF_MAKE_TZONE_ALLOCATED_IMPL(InspectorHeapAgent);

InspectorHeapAgent::InspectorHeapAgent(AgentContext& context)
    : InspectorAgentBase("Heap"_s)
    , m_injectedScriptManager(context.injectedScriptManager)
    , m_frontendDispatcher(makeUnique<HeapFrontendDispatcher>(context.frontendRouter))
    , m_backendDispatcher(HeapBackendDispatcher::create(context.backendDispatcher, this))
    , m_environment(context.environment)
{
}

InspectorHeapAgent::~InspectorHeapAgent() = default;

void InspectorHeapAgent::didCreateFrontendAndBackend(FrontendRouter*, BackendDispatcher*)
{
}

void InspectorHeapAgent::willDestroyFrontendAndBackend(DisconnectReason)
{
    disable();
}

Protocol::ErrorStringOr<void> InspectorHeapAgent::enable()
{
    if (m_enabled)
        return makeUnexpected("Heap domain already enabled"_s);

    m_enabled = true;

    m_environment.vm().heap.addObserver(this);

    return { };
}

Protocol::ErrorStringOr<void> InspectorHeapAgent::disable()
{
    if (!m_enabled)
        return makeUnexpected("Heap domain already disabled"_s);

    m_enabled = false;
    m_tracking = false;

    m_environment.vm().heap.removeObserver(this);

    clearHeapSnapshots();

    return { };
}

Protocol::ErrorStringOr<void> InspectorHeapAgent::gc()
{
    VM& vm = m_environment.vm();
    JSLockHolder lock(vm);
    sanitizeStackForVM(vm);
    vm.heap.collectNow(Sync, CollectionScope::Full);

    return { };
}

Protocol::ErrorStringOr<std::tuple<double, Protocol::Heap::HeapSnapshotData>> InspectorHeapAgent::snapshot()
{
    VM& vm = m_environment.vm();
    JSLockHolder lock(vm);

    HeapSnapshotBuilder snapshotBuilder(vm.ensureHeapProfiler());
    snapshotBuilder.buildSnapshot();

    auto timestamp = m_environment.executionStopwatch().elapsedTime().seconds();
    auto snapshotData = snapshotBuilder.json([&] (const HeapSnapshotNode& node) {
        if (Structure* structure = node.cell->structure()) {
            if (JSGlobalObject* globalObject = structure->globalObject()) {
                if (!m_environment.canAccessInspectedScriptState(globalObject))
                    return false;
            }
        }
        return true;
    });

    return { { timestamp, snapshotData } };
}

Protocol::ErrorStringOr<void> InspectorHeapAgent::startTracking()
{
    if (m_tracking)
        return { };

    m_tracking = true;

    auto result = snapshot();
    if (!result)
        return makeUnexpected(WTFMove(result.error()));

    auto [timestamp, snapshotData] = WTFMove(result.value());
    m_frontendDispatcher->trackingStart(timestamp, snapshotData);

    return { };
}

Protocol::ErrorStringOr<void> InspectorHeapAgent::stopTracking()
{
    if (!m_tracking)
        return { };

    m_tracking = false;

    auto result = snapshot();
    if (!result)
        return makeUnexpected(WTFMove(result.error()));

    auto [timestamp, snapshotData] = WTFMove(result.value());
    m_frontendDispatcher->trackingComplete(timestamp, snapshotData);

    return { };
}

std::optional<HeapSnapshotNode> InspectorHeapAgent::nodeForHeapObjectIdentifier(Protocol::ErrorString& errorString, unsigned heapObjectIdentifier)
{
    HeapProfiler* heapProfiler = m_environment.vm().heapProfiler();
    if (!heapProfiler) {
        errorString = "No heap snapshot"_s;
        return std::nullopt;
    }

    HeapSnapshot* snapshot = heapProfiler->mostRecentSnapshot();
    if (!snapshot) {
        errorString = "No heap snapshot"_s;
        return std::nullopt;
    }

    const std::optional<HeapSnapshotNode> optionalNode = snapshot->nodeForObjectIdentifier(heapObjectIdentifier);
    if (!optionalNode) {
        errorString = "No object for identifier, it may have been collected"_s;
        return std::nullopt;
    }

    return optionalNode;
}

Protocol::ErrorStringOr<std::tuple<String, RefPtr<Protocol::Debugger::FunctionDetails>, RefPtr<Protocol::Runtime::ObjectPreview>>> InspectorHeapAgent::getPreview(int heapObjectId)
{
    Protocol::ErrorString errorString;

    // Prevent the cell from getting collected as we look it up.
    VM& vm = m_environment.vm();
    JSLockHolder lock(vm);
    DeferGC deferGC(vm);

    unsigned heapObjectIdentifier = static_cast<unsigned>(heapObjectId);
    const std::optional<HeapSnapshotNode> optionalNode = nodeForHeapObjectIdentifier(errorString, heapObjectIdentifier);
    if (!optionalNode)
        return makeUnexpected(errorString);

    // String preview.
    JSCell* cell = optionalNode->cell;
    if (cell->isString())
        return { { asString(cell)->tryGetValue(), nullptr, nullptr } };

    // BigInt preview.
    if (cell->isHeapBigInt())
        return { { JSBigInt::tryGetString(vm, asHeapBigInt(cell), 10), nullptr, nullptr } };

    // FIXME: Provide preview information for Internal Objects? CodeBlock, Executable, etc.

    Structure* structure = cell->structure();
    if (!structure)
        return makeUnexpected("Unable to get object details - Structure"_s);

    JSGlobalObject* globalObject = structure->globalObject();
    if (!globalObject)
        return makeUnexpected("Unable to get object details - GlobalObject"_s);

    InjectedScript injectedScript = m_injectedScriptManager.injectedScriptFor(globalObject);
    if (injectedScript.hasNoValue())
        return makeUnexpected("Unable to get object details - InjectedScript"_s);

    // Function preview.
    if (cell->inherits<JSFunction>()) {
        RefPtr<Protocol::Debugger::FunctionDetails> functionDetails;
        injectedScript.functionDetails(errorString, cell, functionDetails);
        if (!functionDetails)
            return makeUnexpected(errorString);
        return { { nullString(), functionDetails, nullptr } };
    }

    // Object preview.
    return { { nullString(), nullptr, injectedScript.previewValue(cell) } };
}

Protocol::ErrorStringOr<Ref<Protocol::Runtime::RemoteObject>> InspectorHeapAgent::getRemoteObject(int heapObjectId, const String& objectGroup)
{
    Protocol::ErrorString errorString;

    // Prevent the cell from getting collected as we look it up.
    VM& vm = m_environment.vm();
    JSLockHolder lock(vm);
    DeferGC deferGC(vm);

    unsigned heapObjectIdentifier = static_cast<unsigned>(heapObjectId);
    const std::optional<HeapSnapshotNode> optionalNode = nodeForHeapObjectIdentifier(errorString, heapObjectIdentifier);
    if (!optionalNode)
        return makeUnexpected(errorString);

    JSCell* cell = optionalNode->cell;
    Structure* structure = cell->structure();
    if (!structure)
        return makeUnexpected("Unable to get object details - Structure"_s);

    JSGlobalObject* globalObject = structure->globalObject();
    if (!globalObject)
        return makeUnexpected("Unable to get object details - GlobalObject"_s);

    InjectedScript injectedScript = m_injectedScriptManager.injectedScriptFor(globalObject);
    if (injectedScript.hasNoValue())
        return makeUnexpected("Unable to get object details - InjectedScript"_s);

    auto object = injectedScript.wrapObject(cell, objectGroup, true);
    if (!object)
        return makeUnexpected("Internal error: unable to cast Object"_s);

    return object.releaseNonNull();
}

static Protocol::Heap::GarbageCollection::Type protocolTypeForHeapOperation(CollectionScope scope)
{
    switch (scope) {
    case CollectionScope::Full:
        return Protocol::Heap::GarbageCollection::Type::Full;
    case CollectionScope::Eden:
        return Protocol::Heap::GarbageCollection::Type::Partial;
    }
    ASSERT_NOT_REACHED();
    return Protocol::Heap::GarbageCollection::Type::Full;
}

void InspectorHeapAgent::willGarbageCollect()
{
    if (!m_enabled)
        return;

    m_gcStartTime = m_environment.executionStopwatch().elapsedTime();
}

void InspectorHeapAgent::didGarbageCollect(CollectionScope scope)
{
    if (!m_enabled) {
        m_gcStartTime = Seconds::nan();
        return;
    }

    if (m_gcStartTime.isNaN()) {
        // We were not enabled when the GC began.
        return;
    }

    // FIXME: Include number of bytes freed by collection.

    Seconds endTime = m_environment.executionStopwatch().elapsedTime();
    dispatchGarbageCollectedEvent(protocolTypeForHeapOperation(scope), m_gcStartTime, endTime);

    m_gcStartTime = Seconds::nan();
}

void InspectorHeapAgent::clearHeapSnapshots()
{
    VM& vm = m_environment.vm();
    JSLockHolder lock(vm);

    if (HeapProfiler* heapProfiler = vm.heapProfiler()) {
        heapProfiler->clearSnapshots();
        HeapSnapshotBuilder::resetNextAvailableObjectIdentifier();
    }
}

void InspectorHeapAgent::dispatchGarbageCollectedEvent(Protocol::Heap::GarbageCollection::Type type, Seconds startTime, Seconds endTime)
{
    auto protocolObject = Protocol::Heap::GarbageCollection::create()
        .setType(type)
        .setStartTime(startTime.seconds())
        .setEndTime(endTime.seconds())
        .release();

    m_frontendDispatcher->garbageCollected(WTFMove(protocolObject));
}

} // namespace Inspector

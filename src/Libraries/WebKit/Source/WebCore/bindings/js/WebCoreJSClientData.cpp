/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 27, 2025.
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
#include "WebCoreJSClientData.h"

#include "DOMGCOutputConstraint.h"
#include "DocumentInlines.h"
#include "ExtendedDOMClientIsoSubspaces.h"
#include "ExtendedDOMIsoSubspaces.h"
#include "JSAudioWorkletGlobalScope.h"
#include "JSDOMBinding.h"
#include "JSDOMBuiltinConstructorBase.h"
#include "JSDOMWindow.h"
#include "JSDOMWindowProperties.h"
#include "JSDedicatedWorkerGlobalScope.h"
#include "JSIDBSerializationGlobalObject.h"
#include "JSObservableArray.h"
#include "JSPaintWorkletGlobalScope.h"
#include "JSServiceWorkerGlobalScope.h"
#include "JSShadowRealmGlobalScope.h"
#include "JSSharedWorkerGlobalScope.h"
#include "JSWindowProxy.h"
#include "JSWorkerGlobalScope.h"
#include "JSWorkletGlobalScope.h"
#include <JavaScriptCore/FastMallocAlignedMemoryAllocator.h>
#include <JavaScriptCore/HeapInlines.h>
#include <JavaScriptCore/IsoHeapCellType.h>
#include <JavaScriptCore/JSDestructibleObjectHeapCellType.h>
#include <JavaScriptCore/MarkingConstraint.h>
#include <JavaScriptCore/SubspaceInlines.h>
#include <JavaScriptCore/VM.h>
#include "runtime_array.h"
#include "runtime_method.h"
#include "runtime_object.h"
#include <mutex>
#include <wtf/MainThread.h>

namespace WebCore {
using namespace JSC;

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(JSHeapData);

JSHeapData::JSHeapData(Heap& heap)
    : m_runtimeArrayHeapCellType(JSC::IsoHeapCellType::Args<RuntimeArray>())
    , m_observableArrayHeapCellType(JSC::IsoHeapCellType::Args<JSObservableArray>())
    , m_runtimeObjectHeapCellType(JSC::IsoHeapCellType::Args<JSC::Bindings::RuntimeObject>())
    , m_windowProxyHeapCellType(JSC::IsoHeapCellType::Args<JSWindowProxy>())
    , m_heapCellTypeForJSDOMWindow(JSC::IsoHeapCellType::Args<JSDOMWindow>())
    , m_heapCellTypeForJSDedicatedWorkerGlobalScope(JSC::IsoHeapCellType::Args<JSDedicatedWorkerGlobalScope>())
    , m_heapCellTypeForJSWorkerGlobalScope(JSC::IsoHeapCellType::Args<JSWorkerGlobalScope>())
    , m_heapCellTypeForJSSharedWorkerGlobalScope(JSC::IsoHeapCellType::Args<JSSharedWorkerGlobalScope>())
    , m_heapCellTypeForJSShadowRealmGlobalScope(JSC::IsoHeapCellType::Args<JSShadowRealmGlobalScope>())
    , m_heapCellTypeForJSServiceWorkerGlobalScope(JSC::IsoHeapCellType::Args<JSServiceWorkerGlobalScope>())
    , m_heapCellTypeForJSWorkletGlobalScope(JSC::IsoHeapCellType::Args<JSWorkletGlobalScope>())
    , m_heapCellTypeForJSPaintWorkletGlobalScope(JSC::IsoHeapCellType::Args<JSPaintWorkletGlobalScope>())
#if ENABLE(WEB_AUDIO)
    , m_heapCellTypeForJSAudioWorkletGlobalScope(JSC::IsoHeapCellType::Args<JSAudioWorkletGlobalScope>())
#endif
    , m_heapCellTypeForJSIDBSerializationGlobalObject(JSC::IsoHeapCellType::Args<JSIDBSerializationGlobalObject>())
    , m_domBuiltinConstructorSpace ISO_SUBSPACE_INIT(heap, heap.cellHeapCellType, JSDOMBuiltinConstructorBase)
    , m_domConstructorSpace ISO_SUBSPACE_INIT(heap, heap.cellHeapCellType, JSDOMConstructorBase)
    , m_domNamespaceObjectSpace ISO_SUBSPACE_INIT(heap, heap.cellHeapCellType, JSDOMObject)
    , m_domWindowPropertiesSpace ISO_SUBSPACE_INIT(heap, heap.cellHeapCellType, JSDOMWindowProperties)
    , m_runtimeArraySpace ISO_SUBSPACE_INIT(heap, m_runtimeArrayHeapCellType, RuntimeArray)
    , m_observableArraySpace ISO_SUBSPACE_INIT(heap, m_observableArrayHeapCellType, JSObservableArray)
    , m_runtimeMethodSpace ISO_SUBSPACE_INIT(heap, heap.cellHeapCellType, RuntimeMethod) // Hash:0xf70c4a85
    , m_runtimeObjectSpace ISO_SUBSPACE_INIT(heap, m_runtimeObjectHeapCellType, JSC::Bindings::RuntimeObject)
    , m_windowProxySpace ISO_SUBSPACE_INIT(heap, m_windowProxyHeapCellType, JSWindowProxy)
    , m_idbSerializationSpace ISO_SUBSPACE_INIT(heap, m_heapCellTypeForJSIDBSerializationGlobalObject, JSIDBSerializationGlobalObject)
    , m_subspaces(makeUnique<ExtendedDOMIsoSubspaces>())
{
}

JSHeapData* JSHeapData::ensureHeapData(Heap& heap)
{
    if (!Options::useGlobalGC())
        return new JSHeapData(heap);

    static JSHeapData* singleton = nullptr;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [&] {
        singleton = new JSHeapData(heap);
    });
    return singleton;
}

#define CLIENT_ISO_SUBSPACE_INIT(subspace) subspace(m_heapData->subspace)

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(JSVMClientData);

JSVMClientData::JSVMClientData(VM& vm)
    : m_builtinFunctions(vm)
    , m_builtinNames(vm)
    , m_heapData(JSHeapData::ensureHeapData(vm.heap))
    , CLIENT_ISO_SUBSPACE_INIT(m_domBuiltinConstructorSpace)
    , CLIENT_ISO_SUBSPACE_INIT(m_domConstructorSpace)
    , CLIENT_ISO_SUBSPACE_INIT(m_domNamespaceObjectSpace)
    , CLIENT_ISO_SUBSPACE_INIT(m_domWindowPropertiesSpace)
    , CLIENT_ISO_SUBSPACE_INIT(m_runtimeArraySpace)
    , CLIENT_ISO_SUBSPACE_INIT(m_observableArraySpace)
    , CLIENT_ISO_SUBSPACE_INIT(m_runtimeMethodSpace)
    , CLIENT_ISO_SUBSPACE_INIT(m_runtimeObjectSpace)
    , CLIENT_ISO_SUBSPACE_INIT(m_windowProxySpace)
    , CLIENT_ISO_SUBSPACE_INIT(m_idbSerializationSpace)
    , m_clientSubspaces(makeUnique<ExtendedDOMClientIsoSubspaces>())
{
}

#undef CLIENT_ISO_SUBSPACE_INIT

JSVMClientData::~JSVMClientData()
{
    m_clients.forEach([](auto& client) {
        client.willDestroyVM();
    });

    ASSERT(m_worldSet.contains(m_normalWorld.get()));
    ASSERT(m_worldSet.size() == 1);
    ASSERT(m_normalWorld->hasOneRef());
    m_normalWorld = nullptr;
    ASSERT(m_worldSet.isEmpty());
}

void JSVMClientData::getAllWorlds(Vector<Ref<DOMWrapperWorld>>& worlds)
{
    ASSERT(worlds.isEmpty());

    worlds.reserveInitialCapacity(m_worldSet.size());

    // It is necessary to order the `DOMWrapperWorld`s because certain callers expect the main world
    // to be the first item in the list, as they use the main world as an indicator of when the page
    // is ready to start evaluating JavaScript. For example, Web Inspector waits for the main world
    // change to clear any injected scripts and debugger/breakpoint state.

    auto& mainNormalWorld = mainThreadNormalWorld();

    // Add main normal world.
    if (m_worldSet.contains(&mainNormalWorld))
        worlds.append(mainNormalWorld);

    // Add other normal worlds.
    for (auto* world : m_worldSet) {
        if (world->type() != DOMWrapperWorld::Type::Normal)
            continue;
        if (world == &mainNormalWorld)
            continue;
        worlds.append(*world);
    }

    // Add non-normal worlds.
    for (auto* world : m_worldSet) {
        if (world->type() == DOMWrapperWorld::Type::Normal)
            continue;
        worlds.append(*world);
    }
}

void JSVMClientData::initNormalWorld(VM* vm, WorkerThreadType type)
{
    JSVMClientData* clientData = new JSVMClientData(*vm);
    vm->clientData = clientData; // ~VM deletes this pointer.

    vm->heap.addMarkingConstraint(makeUnique<DOMGCOutputConstraint>(*vm, clientData->heapData()));

    clientData->m_normalWorld = DOMWrapperWorld::create(*vm, DOMWrapperWorld::Type::Normal);
    vm->m_typedArrayController = adoptRef(new WebCoreTypedArrayController(type == WorkerThreadType::DedicatedWorker || type == WorkerThreadType::Worklet));
}

String JSVMClientData::overrideSourceURL(const JSC::StackFrame& frame, const String& originalSourceURL) const
{
    if (originalSourceURL.isEmpty())
        return nullString();

    auto* codeBlock = frame.codeBlock();
    RELEASE_ASSERT(codeBlock);

    auto* globalObject = codeBlock->globalObject();
    if (!globalObject->inherits<JSDOMWindowBase>())
        return nullString();

    auto* document = jsCast<const JSDOMWindowBase*>(globalObject)->wrapped().documentIfLocal();
    if (!document)
        return nullString();

    return document->maskedURLForBindingsIfNeeded(URL(originalSourceURL)).string();
}

} // namespace WebCore


/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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
#include "JSWorkerGlobalScopeBase.h"

#include "DOMWrapperWorld.h"
#include "EventLoop.h"
#include "JSDOMExceptionHandling.h"
#include "JSDOMGuardedObject.h"
#include "JSExecState.h"
#include "JSTrustedScript.h"
#include "TrustedType.h"
#include "WorkerGlobalScope.h"
#include "WorkerThread.h"
#include <JavaScriptCore/GlobalObjectMethodTable.h>
#include <JavaScriptCore/JSCInlines.h>
#include <JavaScriptCore/JSCJSValueInlines.h>
#include <JavaScriptCore/JSGlobalProxy.h>
#include <JavaScriptCore/Microtask.h>
#include <wtf/Language.h>

namespace WebCore {
using namespace JSC;

const ClassInfo JSWorkerGlobalScopeBase::s_info = { "WorkerGlobalScope"_s, &JSDOMGlobalObject::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSWorkerGlobalScopeBase) };

const GlobalObjectMethodTable* JSWorkerGlobalScopeBase::globalObjectMethodTable()
{
    static constexpr GlobalObjectMethodTable table = {
        &supportsRichSourceInfo,
        &shouldInterruptScript,
        &javaScriptRuntimeFlags,
        &queueMicrotaskToEventLoop,
        &shouldInterruptScriptBeforeTimeout,
        &moduleLoaderImportModule,
        &moduleLoaderResolve,
        &moduleLoaderFetch,
        &moduleLoaderCreateImportMetaProperties,
        &moduleLoaderEvaluate,
        &promiseRejectionTracker,
        &reportUncaughtExceptionAtEventLoop,
        &currentScriptExecutionOwner,
        &scriptExecutionStatus,
        &reportViolationForUnsafeEval,
        [] { return defaultLanguage(); },
#if ENABLE(WEBASSEMBLY)
        &compileStreaming,
        &instantiateStreaming,
#else
        nullptr,
        nullptr,
#endif
        deriveShadowRealmGlobalObject,
        codeForEval,
        canCompileStrings,
        trustedScriptStructure,
    };
    return &table;
};

JSWorkerGlobalScopeBase::JSWorkerGlobalScopeBase(JSC::VM& vm, JSC::Structure* structure, RefPtr<WorkerGlobalScope>&& impl)
    : JSDOMGlobalObject(vm, structure, normalWorld(vm), globalObjectMethodTable())
    , m_wrapped(WTFMove(impl))
{
}

void JSWorkerGlobalScopeBase::finishCreation(VM& vm, JSGlobalProxy* proxy)
{
    m_proxy.set(vm, this, proxy);

    Base::finishCreation(vm, m_proxy.get());
    ASSERT(inherits(info()));
}

template<typename Visitor>
void JSWorkerGlobalScopeBase::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    JSWorkerGlobalScopeBase* thisObject = jsCast<JSWorkerGlobalScopeBase*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
    visitor.append(thisObject->m_proxy);
}

DEFINE_VISIT_CHILDREN(JSWorkerGlobalScopeBase);

void JSWorkerGlobalScopeBase::destroy(JSCell* cell)
{
    static_cast<JSWorkerGlobalScopeBase*>(cell)->JSWorkerGlobalScopeBase::~JSWorkerGlobalScopeBase();
}

ScriptExecutionContext* JSWorkerGlobalScopeBase::scriptExecutionContext() const
{
    return m_wrapped.get();
}

bool JSWorkerGlobalScopeBase::supportsRichSourceInfo(const JSGlobalObject* object)
{
    return JSGlobalObject::supportsRichSourceInfo(object);
}

bool JSWorkerGlobalScopeBase::shouldInterruptScript(const JSGlobalObject* object)
{
    return JSGlobalObject::shouldInterruptScript(object);
}

bool JSWorkerGlobalScopeBase::shouldInterruptScriptBeforeTimeout(const JSGlobalObject* object)
{
    return JSGlobalObject::shouldInterruptScriptBeforeTimeout(object);
}

RuntimeFlags JSWorkerGlobalScopeBase::javaScriptRuntimeFlags(const JSGlobalObject* object)
{
    const JSWorkerGlobalScopeBase *thisObject = jsCast<const JSWorkerGlobalScopeBase*>(object);
    return thisObject->m_wrapped->thread().runtimeFlags();
}

JSC::ScriptExecutionStatus JSWorkerGlobalScopeBase::scriptExecutionStatus(JSC::JSGlobalObject* globalObject, JSC::JSObject* owner)
{
    ASSERT_UNUSED(owner, globalObject == owner);
    return jsCast<JSWorkerGlobalScopeBase*>(globalObject)->scriptExecutionContext()->jscScriptExecutionStatus();
}

void JSWorkerGlobalScopeBase::reportViolationForUnsafeEval(JSC::JSGlobalObject* globalObject, const String& source)
{
    return JSGlobalObject::reportViolationForUnsafeEval(globalObject, source);
}

void JSWorkerGlobalScopeBase::queueMicrotaskToEventLoop(JSGlobalObject& object, JSC::QueuedTask&& task)
{
    JSWorkerGlobalScopeBase& thisObject = static_cast<JSWorkerGlobalScopeBase&>(object);
    auto& context = thisObject.wrapped();
    task.setDispatcher(context.eventLoop().jsMicrotaskDispatcher());
    context.eventLoop().queueMicrotask(WTFMove(task));
}

JSValue toJS(JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject*, WorkerGlobalScope& workerGlobalScope)
{
    return toJS(lexicalGlobalObject, workerGlobalScope);
}

JSValue toJS(JSGlobalObject*, WorkerGlobalScope& workerGlobalScope)
{
    auto* script = workerGlobalScope.script();
    if (!script)
        return jsNull();
    auto* contextWrapper = script->globalScopeWrapper();
    ASSERT(contextWrapper);
    return &contextWrapper->proxy();
}

} // namespace WebCore

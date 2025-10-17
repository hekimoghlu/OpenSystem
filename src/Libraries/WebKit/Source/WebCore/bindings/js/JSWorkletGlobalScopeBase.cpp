/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 15, 2024.
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
#include "JSWorkletGlobalScopeBase.h"

#include "DOMWrapperWorld.h"
#include "JSDOMGuardedObject.h"
#include "WorkerOrWorkletScriptController.h"
#include "WorkletGlobalScope.h"
#include <JavaScriptCore/GlobalObjectMethodTable.h>
#include <JavaScriptCore/JSCInlines.h>
#include <JavaScriptCore/JSCJSValueInlines.h>
#include <wtf/Language.h>

namespace WebCore {

using namespace JSC;

const ClassInfo JSWorkletGlobalScopeBase::s_info = { "WorkletGlobalScope"_s, &JSDOMGlobalObject::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSWorkletGlobalScopeBase) };

const GlobalObjectMethodTable* JSWorkletGlobalScopeBase::globalObjectMethodTable()
{
    static constexpr GlobalObjectMethodTable table = {
        &supportsRichSourceInfo,
        &shouldInterruptScript,
        &javaScriptRuntimeFlags,
        nullptr, // queueMicrotaskToEventLoop
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

JSWorkletGlobalScopeBase::JSWorkletGlobalScopeBase(JSC::VM& vm, JSC::Structure* structure, RefPtr<WorkletGlobalScope>&& impl)
    : JSDOMGlobalObject(vm, structure, normalWorld(vm), globalObjectMethodTable())
    , m_wrapped(WTFMove(impl))
{
}

void JSWorkletGlobalScopeBase::finishCreation(VM& vm, JSGlobalProxy* proxy)
{
    m_proxy.set(vm, this, proxy);

    Base::finishCreation(vm, m_proxy.get());
    ASSERT(inherits(info()));
}

template<typename Visitor>
void JSWorkletGlobalScopeBase::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    auto* thisObject = jsCast<JSWorkletGlobalScopeBase*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
    visitor.append(thisObject->m_proxy);
}

DEFINE_VISIT_CHILDREN(JSWorkletGlobalScopeBase);

void JSWorkletGlobalScopeBase::destroy(JSCell* cell)
{
    static_cast<JSWorkletGlobalScopeBase*>(cell)->JSWorkletGlobalScopeBase::~JSWorkletGlobalScopeBase();
}

ScriptExecutionContext* JSWorkletGlobalScopeBase::scriptExecutionContext() const
{
    return m_wrapped.get();
}

JSC::ScriptExecutionStatus JSWorkletGlobalScopeBase::scriptExecutionStatus(JSC::JSGlobalObject* globalObject, JSC::JSObject* owner)
{
    ASSERT_UNUSED(owner, globalObject == owner);
    return jsCast<JSWorkletGlobalScopeBase*>(globalObject)->scriptExecutionContext()->jscScriptExecutionStatus();
}

void JSWorkletGlobalScopeBase::reportViolationForUnsafeEval(JSC::JSGlobalObject* globalObject, const String& source)
{
    return JSGlobalObject::reportViolationForUnsafeEval(globalObject, source);
}

bool JSWorkletGlobalScopeBase::supportsRichSourceInfo(const JSGlobalObject* object)
{
    return JSGlobalObject::supportsRichSourceInfo(object);
}

bool JSWorkletGlobalScopeBase::shouldInterruptScript(const JSGlobalObject* object)
{
    return JSGlobalObject::shouldInterruptScript(object);
}

bool JSWorkletGlobalScopeBase::shouldInterruptScriptBeforeTimeout(const JSGlobalObject* object)
{
    return JSGlobalObject::shouldInterruptScriptBeforeTimeout(object);
}

RuntimeFlags JSWorkletGlobalScopeBase::javaScriptRuntimeFlags(const JSGlobalObject* object)
{
    auto* thisObject = jsCast<const JSWorkletGlobalScopeBase*>(object);
    return thisObject->m_wrapped->jsRuntimeFlags();
}

JSValue toJS(JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject*, WorkletGlobalScope& workletGlobalScope)
{
    return toJS(lexicalGlobalObject, workletGlobalScope);
}

JSValue toJS(JSGlobalObject*, WorkletGlobalScope& workletGlobalScope)
{
    if (!workletGlobalScope.script())
        return jsUndefined();
    auto* contextWrapper = workletGlobalScope.script()->globalScopeWrapper();
    if (!contextWrapper)
        return jsUndefined();
    return &contextWrapper->proxy();
}

} // namespace WebCore

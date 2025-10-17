/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 24, 2024.
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
#include "JSShadowRealmGlobalScopeBase.h"

#include "EventLoop.h"
#include "JSShadowRealmGlobalScope.h"
#include "ScriptModuleLoader.h"
#include "ShadowRealmGlobalScope.h"
#include <JavaScriptCore/GlobalObjectMethodTable.h>
#include <JavaScriptCore/JSCInlines.h>
#include <JavaScriptCore/JSCJSValueInlines.h>
#include <JavaScriptCore/JSGlobalProxy.h>
#include <JavaScriptCore/Microtask.h>
#include <wtf/Language.h>

namespace WebCore {

using namespace JSC;

const ClassInfo JSShadowRealmGlobalScopeBase::s_info = { "ShadowRealmGlobalScope"_s, &JSDOMGlobalObject::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSShadowRealmGlobalScopeBase) };

const GlobalObjectMethodTable* JSShadowRealmGlobalScopeBase::globalObjectMethodTable()
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
        &deriveShadowRealmGlobalObject,
        &codeForEval,
        &canCompileStrings,
        &trustedScriptStructure,
    };
    return &table;
};

JSShadowRealmGlobalScopeBase::JSShadowRealmGlobalScopeBase(JSC::VM& vm, JSC::Structure* structure, RefPtr<ShadowRealmGlobalScope>&& impl)
    : JSDOMGlobalObject(vm, structure, normalWorld(vm), globalObjectMethodTable())
    , m_wrapped(WTFMove(impl))
{
}

void JSShadowRealmGlobalScopeBase::finishCreation(VM& vm, JSGlobalProxy* proxy)
{
    m_proxy.set(vm, this, proxy);
    m_wrapped->m_wrapper = JSC::Weak(this);
    Base::finishCreation(vm, m_proxy.get());
    ASSERT(inherits(info()));
}

template<typename Visitor>
void JSShadowRealmGlobalScopeBase::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    JSShadowRealmGlobalScopeBase* thisObject = jsCast<JSShadowRealmGlobalScopeBase*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
    visitor.append(thisObject->m_proxy);
}

DEFINE_VISIT_CHILDREN(JSShadowRealmGlobalScopeBase);

ScriptExecutionContext* JSShadowRealmGlobalScopeBase::scriptExecutionContext() const
{
    return incubatingRealm()->scriptExecutionContext();
}

const JSDOMGlobalObject* JSShadowRealmGlobalScopeBase::incubatingRealm() const
{
    auto incubatingWrapper = m_wrapped->m_incubatingWrapper.get();
    ASSERT(incubatingWrapper);
    return incubatingWrapper;
}

void JSShadowRealmGlobalScopeBase::destroy(JSCell* cell)
{
    static_cast<JSShadowRealmGlobalScopeBase*>(cell)->JSShadowRealmGlobalScopeBase::~JSShadowRealmGlobalScopeBase();
}

bool JSShadowRealmGlobalScopeBase::supportsRichSourceInfo(const JSGlobalObject* object)
{
    auto incubating = jsCast<const JSShadowRealmGlobalScopeBase*>(object)->incubatingRealm();
    return incubating->globalObjectMethodTable()->supportsRichSourceInfo(incubating);
}

bool JSShadowRealmGlobalScopeBase::shouldInterruptScript(const JSGlobalObject* object)
{
    auto incubating = jsCast<const JSShadowRealmGlobalScopeBase*>(object)->incubatingRealm();
    return incubating->globalObjectMethodTable()->shouldInterruptScript(incubating);
}

bool JSShadowRealmGlobalScopeBase::shouldInterruptScriptBeforeTimeout(const JSGlobalObject* object)
{
    auto incubating = jsCast<const JSShadowRealmGlobalScopeBase*>(object)->incubatingRealm();
    return incubating->globalObjectMethodTable()->shouldInterruptScriptBeforeTimeout(incubating);
}

RuntimeFlags JSShadowRealmGlobalScopeBase::javaScriptRuntimeFlags(const JSGlobalObject* object)
{
    auto incubating = jsCast<const JSShadowRealmGlobalScopeBase*>(object)->incubatingRealm();
    return incubating->globalObjectMethodTable()->javaScriptRuntimeFlags(incubating);
}

JSC::ScriptExecutionStatus JSShadowRealmGlobalScopeBase::scriptExecutionStatus(JSC::JSGlobalObject* globalObject, JSC::JSObject* owner)
{
    auto incubating = jsCast<JSShadowRealmGlobalScopeBase*>(globalObject)->incubatingRealm();
    return incubating->globalObjectMethodTable()->scriptExecutionStatus(incubating, owner);
}

void JSShadowRealmGlobalScopeBase::reportViolationForUnsafeEval(JSC::JSGlobalObject* globalObject, const String& msg)
{
    auto incubating = jsCast<JSShadowRealmGlobalScopeBase*>(globalObject)->incubatingRealm();
    incubating->globalObjectMethodTable()->reportViolationForUnsafeEval(incubating, msg);
}

void JSShadowRealmGlobalScopeBase::queueMicrotaskToEventLoop(JSGlobalObject& object, QueuedTask&& task)
{
    auto incubating = jsCast<JSShadowRealmGlobalScopeBase*>(&object)->incubatingRealm();
    incubating->globalObjectMethodTable()->queueMicrotaskToEventLoop(*incubating, WTFMove(task));
}

JSValue toJS(JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject*, ShadowRealmGlobalScope& realmGlobalScope)
{
    return toJS(lexicalGlobalObject, realmGlobalScope);
}

JSValue toJS(JSGlobalObject*, ShadowRealmGlobalScope& realmGlobalScope)
{
    ASSERT(realmGlobalScope.wrapper());
    return &realmGlobalScope.wrapper()->proxy();
}

} // namespace WebCore

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 5, 2024.
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
#include "InjectedScriptManager.h"

#include "BuiltinNames.h"
#include "CatchScope.h"
#include "InjectedScriptHost.h"
#include "JSLock.h"
#include "JSObjectInlines.h"
#include "SourceCode.h"
#include <wtf/JSONValues.h>
#include <wtf/TZoneMallocInlines.h>

namespace Inspector {

using namespace JSC;

WTF_MAKE_TZONE_ALLOCATED_IMPL(InjectedScriptManager);

InjectedScriptManager::InjectedScriptManager(InspectorEnvironment& environment, Ref<InjectedScriptHost>&& injectedScriptHost)
    : m_environment(environment)
    , m_injectedScriptHost(WTFMove(injectedScriptHost))
    , m_nextInjectedScriptId(1)
{
}

InjectedScriptManager::~InjectedScriptManager() = default;

void InjectedScriptManager::connect()
{
}

void InjectedScriptManager::disconnect()
{
    discardInjectedScripts();
}

void InjectedScriptManager::discardInjectedScripts()
{
    m_injectedScriptHost->clearAllWrappers();
    m_idToInjectedScript.clear();
    m_scriptStateToId.clear();
}

InjectedScriptHost& InjectedScriptManager::injectedScriptHost()
{
    return m_injectedScriptHost.get();
}

InjectedScript InjectedScriptManager::injectedScriptForId(int id)
{
    auto it = m_idToInjectedScript.find(id);
    if (it != m_idToInjectedScript.end())
        return it->value;

    for (auto it = m_scriptStateToId.begin(); it != m_scriptStateToId.end(); ++it) {
        if (it->value == id)
            return injectedScriptFor(it->key);
    }

    return InjectedScript();
}

int InjectedScriptManager::injectedScriptIdFor(JSGlobalObject* globalObject)
{
    auto it = m_scriptStateToId.find(globalObject);
    if (it != m_scriptStateToId.end())
        return it->value;

    int id = m_nextInjectedScriptId++;
    m_scriptStateToId.set(globalObject, id);
    return id;
}

InjectedScript InjectedScriptManager::injectedScriptForObjectId(const String& objectId)
{
    auto parsedObjectId = JSON::Value::parseJSON(objectId);
    if (!parsedObjectId)
        return InjectedScript();

    auto resultObject = parsedObjectId->asObject();
    if (!resultObject)
        return InjectedScript();

    auto injectedScriptId = resultObject->getInteger("injectedScriptId"_s);
    if (!injectedScriptId)
        return InjectedScript();

    return m_idToInjectedScript.get(*injectedScriptId);
}

void InjectedScriptManager::releaseObjectGroup(const String& objectGroup)
{
    for (auto& injectedScript : m_idToInjectedScript.values())
        injectedScript.releaseObjectGroup(objectGroup);
}

void InjectedScriptManager::clearEventValue()
{
    for (auto& injectedScript : m_idToInjectedScript.values())
        injectedScript.clearEventValue();
}

void InjectedScriptManager::clearExceptionValue()
{
    for (auto& injectedScript : m_idToInjectedScript.values())
        injectedScript.clearExceptionValue();
}

Expected<JSObject*, NakedPtr<Exception>> InjectedScriptManager::createInjectedScript(JSGlobalObject* globalObject, int id)
{
    VM& vm = globalObject->vm();
    JSLockHolder lock(vm);
    auto scope = DECLARE_CATCH_SCOPE(vm);

    JSValue globalThisValue = globalObject->globalThis();

    auto* functionValue = jsCast<JSFunction*>(globalObject->linkTimeConstant(LinkTimeConstant::createInspectorInjectedScript));
    RETURN_IF_EXCEPTION(scope, makeUnexpected(scope.exception()));
    if (!functionValue)
        return nullptr;

    auto callData = JSC::getCallData(functionValue);
    if (callData.type == CallData::Type::None)
        return nullptr;

    MarkedArgumentBuffer args;
    args.append(m_injectedScriptHost->wrapper(globalObject));
    args.append(globalThisValue);
    args.append(jsNumber(id));
    ASSERT(!args.hasOverflowed());

    JSValue result = JSC::call(globalObject, functionValue, callData, globalThisValue, args);
    RETURN_IF_EXCEPTION(scope, makeUnexpected(scope.exception()));
    return result.getObject();
}

InjectedScript InjectedScriptManager::injectedScriptFor(JSGlobalObject* globalObject)
{
    auto it = m_scriptStateToId.find(globalObject);
    if (it != m_scriptStateToId.end()) {
        auto it1 = m_idToInjectedScript.find(it->value);
        if (it1 != m_idToInjectedScript.end())
            return it1->value;
    }

    if (!m_environment.canAccessInspectedScriptState(globalObject))
        return InjectedScript();

    int id = injectedScriptIdFor(globalObject);
    auto createResult = createInjectedScript(globalObject, id);
    if (!createResult) {
        auto& error = createResult.error();
        ASSERT(error);

        if (globalObject->vm().isTerminationException(error))
            return InjectedScript();

        LineColumn lineColumn;
        auto& stack = error->stack();
        if (stack.size() > 0)
            lineColumn = stack[0].computeLineAndColumn();
        WTFLogAlways("Error when creating injected script: %s (%d:%d)\n", error->value().toWTFString(globalObject).utf8().data(), lineColumn.line, lineColumn.column);
        RELEASE_ASSERT_NOT_REACHED();
    }
    if (!createResult.value()) {
        WTFLogAlways("Missing injected script object");
        RELEASE_ASSERT_NOT_REACHED();
    }

    InjectedScript result(globalObject, createResult.value(), &m_environment);
    m_idToInjectedScript.set(id, result);
    didCreateInjectedScript(result);
    return result;
}

void InjectedScriptManager::didCreateInjectedScript(const InjectedScript&)
{
    // Intentionally empty. This allows for subclasses to inject additional scripts.
}

} // namespace Inspector


/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 23, 2023.
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
#include "CommandLineAPIHost.h"

#include "Database.h"
#include "Document.h"
#include "EventTarget.h"
#include "InspectorDOMStorageAgent.h"
#include "InspectorDatabaseAgent.h"
#include "JSCommandLineAPIHost.h"
#include "JSDOMGlobalObject.h"
#include "JSEventListener.h"
#include "PagePasteboardContext.h"
#include "Pasteboard.h"
#include "Storage.h"
#include "WebConsoleAgent.h"
#include <JavaScriptCore/InjectedScriptBase.h>
#include <JavaScriptCore/InspectorAgent.h>
#include <JavaScriptCore/JSCInlines.h>
#include <JavaScriptCore/JSLock.h>
#include <JavaScriptCore/ObjectConstructor.h>
#include <wtf/JSONValues.h>
#include <wtf/RefPtr.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(WEB_RTC)
#include "JSRTCPeerConnection.h"
#include "RTCLogsCallback.h"
#endif

namespace WebCore {

using namespace JSC;
using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(CommandLineAPIHost::InspectableObject);

Ref<CommandLineAPIHost> CommandLineAPIHost::create()
{
    return adoptRef(*new CommandLineAPIHost);
}

CommandLineAPIHost::CommandLineAPIHost()
    : m_inspectedObject(makeUnique<InspectableObject>())
{
}

CommandLineAPIHost::~CommandLineAPIHost() = default;

void CommandLineAPIHost::disconnect()
{

    m_instrumentingAgents = nullptr;
}

void CommandLineAPIHost::inspect(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue object, JSC::JSValue hints)
{
    if (!m_instrumentingAgents)
        return;

    auto* inspectorAgent = m_instrumentingAgents->persistentInspectorAgent();
    if (!inspectorAgent)
        return;

    auto objectValue = Inspector::toInspectorValue(&lexicalGlobalObject, object);
    if (!objectValue)
        return;

    auto hintsValue = Inspector::toInspectorValue(&lexicalGlobalObject, hints);
    if (!hintsValue)
        return;

    auto hintsObject = hintsValue->asObject();
    if (!hintsObject)
        return;

    auto remoteObject = Inspector::Protocol::BindingTraits<Inspector::Protocol::Runtime::RemoteObject>::runtimeCast(objectValue.releaseNonNull());
    inspectorAgent->inspect(WTFMove(remoteObject), hintsObject.releaseNonNull());
}

CommandLineAPIHost::EventListenersRecord CommandLineAPIHost::getEventListeners(JSGlobalObject& lexicalGlobalObject, EventTarget& target)
{
    auto* scriptExecutionContext = target.scriptExecutionContext();
    if (!scriptExecutionContext)
        return { };

    EventListenersRecord result;

    VM& vm = lexicalGlobalObject.vm();

    for (auto& eventType : target.eventTypes()) {
        Vector<CommandLineAPIHost::ListenerEntry> entries;

        for (auto& eventListener : target.eventListeners(eventType)) {
            if (!is<JSEventListener>(eventListener->callback()))
                continue;

            auto& jsListener = downcast<JSEventListener>(eventListener->callback());

            // Hide listeners from other contexts.
            if (jsListener.isolatedWorld() != &currentWorld(lexicalGlobalObject))
                continue;

            auto* function = jsListener.ensureJSFunction(*scriptExecutionContext);
            if (!function)
                continue;

            entries.append({ Strong<JSObject>(vm, function), eventListener->useCapture(), eventListener->isPassive(), eventListener->isOnce() });
        }

        if (!entries.isEmpty())
            result.append({ eventType, WTFMove(entries) });
    }

    return result;
}

#if ENABLE(WEB_RTC)
void CommandLineAPIHost::gatherRTCLogs(JSGlobalObject& globalObject, RefPtr<RTCLogsCallback>&& callback)
{
    RefPtr document = dynamicDowncast<Document>(jsCast<JSDOMGlobalObject*>(&globalObject)->scriptExecutionContext());
    if (!document)
        return;

    if (!callback) {
        document->stopGatheringRTCLogs();
        return;
    }

    document->startGatheringRTCLogs([callback = callback.releaseNonNull()] (auto&& logType, auto&& logMessage, auto&& logLevel, auto&& connection) mutable {
        ASSERT(!logType.isNull());
        ASSERT(!logMessage.isNull());

        callback->handleEvent({ WTFMove(logType), WTFMove(logMessage), WTFMove(logLevel), WTFMove(connection) });
    });
}
#endif

void CommandLineAPIHost::copyText(const String& text)
{
    Pasteboard::createForCopyAndPaste({ })->writePlainText(text, Pasteboard::CannotSmartReplace);
}

JSC::JSValue CommandLineAPIHost::InspectableObject::get(JSC::JSGlobalObject&)
{
    return { };
}

void CommandLineAPIHost::addInspectedObject(std::unique_ptr<CommandLineAPIHost::InspectableObject> object)
{
    m_inspectedObject = WTFMove(object);
}

JSC::JSValue CommandLineAPIHost::inspectedObject(JSC::JSGlobalObject& lexicalGlobalObject)
{
    if (!m_inspectedObject)
        return jsUndefined();

    JSC::JSLockHolder lock(&lexicalGlobalObject);
    auto scriptValue = m_inspectedObject->get(lexicalGlobalObject);
    return scriptValue ? scriptValue : jsUndefined();
}

String CommandLineAPIHost::databaseId(Database& database)
{
    if (m_instrumentingAgents) {
        if (auto* databaseAgent = m_instrumentingAgents->enabledDatabaseAgent())
            return databaseAgent->databaseId(database);
    }
    return { };
}

String CommandLineAPIHost::storageId(Storage& storage)
{
    return InspectorDOMStorageAgent::storageId(storage);
}

JSValue CommandLineAPIHost::wrapper(JSGlobalObject* exec, JSDOMGlobalObject* globalObject)
{
    JSValue value = m_wrappers.getWrapper(globalObject);
    if (value)
        return value;

    JSObject* prototype = JSCommandLineAPIHost::createPrototype(exec->vm(), *globalObject);
    Structure* structure = JSCommandLineAPIHost::createStructure(exec->vm(), globalObject, prototype);
    JSCommandLineAPIHost* commandLineAPIHost = JSCommandLineAPIHost::create(structure, globalObject, Ref { *this });
    m_wrappers.addWrapper(globalObject, commandLineAPIHost);

    return commandLineAPIHost;
}

void CommandLineAPIHost::clearAllWrappers()
{
    m_wrappers.clearAllWrappers();
    m_inspectedObject = makeUnique<InspectableObject>();
}

} // namespace WebCore

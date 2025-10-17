/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 30, 2023.
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
#include "WorkerConsoleClient.h"

#include "InspectorInstrumentation.h"
#include "WorkerGlobalScope.h"
#include <JavaScriptCore/ConsoleMessage.h>
#include <JavaScriptCore/ScriptArguments.h>
#include <JavaScriptCore/ScriptCallStack.h>
#include <JavaScriptCore/ScriptCallStackFactory.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WorkerConsoleClient);

WorkerConsoleClient::WorkerConsoleClient(WorkerOrWorkletGlobalScope& globalScope)
    : m_globalScope(globalScope)
{
}

WorkerConsoleClient::~WorkerConsoleClient() = default;

void WorkerConsoleClient::messageWithTypeAndLevel(MessageType type, MessageLevel level, JSC::JSGlobalObject* exec, Ref<Inspector::ScriptArguments>&& arguments)
{
    String messageText;
    arguments->getFirstArgumentAsString(messageText);
    auto message = makeUnique<Inspector::ConsoleMessage>(MessageSource::ConsoleAPI, type, level, messageText, WTFMove(arguments), exec);
    m_globalScope.addConsoleMessage(WTFMove(message));
}

void WorkerConsoleClient::count(JSC::JSGlobalObject* exec, const String& label)
{
    // FIXME: Add support for WorkletGlobalScope.
    if (auto* worker = dynamicDowncast<WorkerGlobalScope>(m_globalScope))
        InspectorInstrumentation::consoleCount(*worker, exec, label);
}

void WorkerConsoleClient::countReset(JSC::JSGlobalObject* exec, const String& label)
{
    // FIXME: Add support for WorkletGlobalScope.
    if (auto* worker = dynamicDowncast<WorkerGlobalScope>(m_globalScope))
        InspectorInstrumentation::consoleCountReset(*worker, exec, label);
}

void WorkerConsoleClient::time(JSC::JSGlobalObject* exec, const String& label)
{
    // FIXME: Add support for WorkletGlobalScope.
    if (auto* worker = dynamicDowncast<WorkerGlobalScope>(m_globalScope))
        InspectorInstrumentation::startConsoleTiming(*worker, exec, label);
}

void WorkerConsoleClient::timeLog(JSC::JSGlobalObject* exec, const String& label, Ref<ScriptArguments>&& arguments)
{
    // FIXME: Add support for WorkletGlobalScope.
    if (auto* worker = dynamicDowncast<WorkerGlobalScope>(m_globalScope))
        InspectorInstrumentation::logConsoleTiming(*worker, exec, label, WTFMove(arguments));
}

void WorkerConsoleClient::timeEnd(JSC::JSGlobalObject* exec, const String& label)
{
    // FIXME: Add support for WorkletGlobalScope.
    if (auto* worker = dynamicDowncast<WorkerGlobalScope>(m_globalScope))
        InspectorInstrumentation::stopConsoleTiming(*worker, exec, label);
}

// FIXME: <https://webkit.org/b/153499> Web Inspector: console.profile should use the new Sampling Profiler
void WorkerConsoleClient::profile(JSC::JSGlobalObject*, const String&) { }
void WorkerConsoleClient::profileEnd(JSC::JSGlobalObject*, const String&) { }

// FIXME: <https://webkit.org/b/127634> Web Inspector: support debugging web workers
void WorkerConsoleClient::takeHeapSnapshot(JSC::JSGlobalObject*, const String&) { }
void WorkerConsoleClient::timeStamp(JSC::JSGlobalObject*, Ref<ScriptArguments>&&) { }

// FIXME: <https://webkit.org/b/243362> Web Inspector: support starting/stopping recordings from the console in a Worker
void WorkerConsoleClient::record(JSC::JSGlobalObject*, Ref<ScriptArguments>&&) { }
void WorkerConsoleClient::recordEnd(JSC::JSGlobalObject*, Ref<ScriptArguments>&&) { }

// FIXME: <https://webkit.org/b/243361> Web Inspector: support console screenshots in a Worker
void WorkerConsoleClient::screenshot(JSC::JSGlobalObject*, Ref<ScriptArguments>&&) { }

} // namespace WebCore

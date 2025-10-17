/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 18, 2022.
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
#include "WorkletGlobalScope.h"

#include "InspectorInstrumentation.h"
#include "JSWorkletGlobalScope.h"
#include "LocalFrame.h"
#include "PageConsoleClient.h"
#include "SecurityOriginPolicy.h"
#include "Settings.h"
#include "WorkerMessagePortChannelProvider.h"
#include "WorkerOrWorkletThread.h"
#include "WorkerScriptLoader.h"
#include "WorkletParameters.h"
#include <JavaScriptCore/Exception.h>
#include <JavaScriptCore/JSLock.h>
#include <JavaScriptCore/ScriptCallStack.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
using namespace Inspector;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WorkletGlobalScope);

static std::atomic<unsigned> gNumberOfWorkletGlobalScopes { 0 };

WorkletGlobalScope::WorkletGlobalScope(WorkerOrWorkletThread& thread, Ref<JSC::VM>&& vm, const WorkletParameters& parameters)
    : WorkerOrWorkletGlobalScope(WorkerThreadType::Worklet, parameters.sessionID, WTFMove(vm), parameters.referrerPolicy, &thread, parameters.noiseInjectionHashSalt, parameters.advancedPrivacyProtections)
    , m_topOrigin(SecurityOrigin::createOpaque())
    , m_url(parameters.windowURL)
    , m_jsRuntimeFlags(parameters.jsRuntimeFlags)
    , m_settingsValues(parameters.settingsValues)
{
    ++gNumberOfWorkletGlobalScopes;

    setStorageBlockingPolicy(parameters.settingsValues.storageBlockingPolicy);
    setSecurityOriginPolicy(SecurityOriginPolicy::create(SecurityOrigin::create(this->url())));
    setContentSecurityPolicy(makeUnique<ContentSecurityPolicy>(URL { this->url() }, *this));
}

WorkletGlobalScope::WorkletGlobalScope(Document& document, Ref<JSC::VM>&& vm, ScriptSourceCode&& code)
    : WorkerOrWorkletGlobalScope(WorkerThreadType::Worklet, *document.sessionID(), WTFMove(vm), document.referrerPolicy(), nullptr, document.noiseInjectionHashSalt(), document.advancedPrivacyProtections())
    , m_document(document)
    , m_topOrigin(SecurityOrigin::createOpaque())
    , m_url(code.url())
    , m_jsRuntimeFlags(document.settings().javaScriptRuntimeFlags())
    , m_code(WTFMove(code))
    , m_settingsValues(document.settingsValues().isolatedCopy())
{
    ++gNumberOfWorkletGlobalScopes;

    ASSERT(document.page());

    setStorageBlockingPolicy(m_document->settings().storageBlockingPolicy());
    setSecurityOriginPolicy(SecurityOriginPolicy::create(SecurityOrigin::create(this->url())));
    setContentSecurityPolicy(makeUnique<ContentSecurityPolicy>(URL { this->url() }, *this));
}

WorkletGlobalScope::~WorkletGlobalScope()
{
    ASSERT(!script());
    removeFromContextsMap();
    ASSERT(gNumberOfWorkletGlobalScopes);
    --gNumberOfWorkletGlobalScopes;
}

unsigned WorkletGlobalScope::numberOfWorkletGlobalScopes()
{
    return gNumberOfWorkletGlobalScopes;
}

void WorkletGlobalScope::prepareForDestruction()
{
    WorkerOrWorkletGlobalScope::prepareForDestruction();

    if (script()) {
        script()->vm().notifyNeedTermination();
        clearScript();
    }
}

String WorkletGlobalScope::userAgent(const URL& url) const
{
    if (!m_document)
        return emptyString();
    return m_document->userAgent(url);
}

void WorkletGlobalScope::evaluate()
{
    if (m_code)
        script()->evaluate(*m_code);
}

URL WorkletGlobalScope::completeURL(const String& url, ForceUTF8) const
{
    if (url.isNull())
        return URL();
    return URL(this->url(), url);
}

void WorkletGlobalScope::logExceptionToConsole(const String& errorMessage, const String& sourceURL, int lineNumber, int columnNumber, RefPtr<ScriptCallStack>&& stack)
{
    if (UNLIKELY(settingsValues().logsPageMessagesToSystemConsoleEnabled)) {
        if (stack) {
            Inspector::ConsoleMessage message { MessageSource::JS, MessageType::Log, MessageLevel::Error, errorMessage, *stack };
            PageConsoleClient::logMessageToSystemConsole(message);
        } else {
            Inspector::ConsoleMessage message { MessageSource::JS, MessageType::Log, MessageLevel::Error, errorMessage, sourceURL, static_cast<unsigned>(lineNumber), static_cast<unsigned>(columnNumber) };
            PageConsoleClient::logMessageToSystemConsole(message);
        }
    }

    if (!m_document || isJSExecutionForbidden())
        return;
    m_document->logExceptionToConsole(errorMessage, sourceURL, lineNumber, columnNumber, WTFMove(stack));
}

void WorkletGlobalScope::addConsoleMessage(std::unique_ptr<Inspector::ConsoleMessage>&& message)
{
    if (UNLIKELY(settingsValues().logsPageMessagesToSystemConsoleEnabled && message))
        PageConsoleClient::logMessageToSystemConsole(*message);

    if (!m_document || isJSExecutionForbidden() || !message)
        return;
    m_document->addConsoleMessage(makeUnique<Inspector::ConsoleMessage>(message->source(), message->type(), message->level(), message->message(), 0));
}

void WorkletGlobalScope::addConsoleMessage(MessageSource source, MessageLevel level, const String& message, unsigned long requestIdentifier)
{
    if (!m_document || isJSExecutionForbidden())
        return;
    m_document->addConsoleMessage(source, level, message, requestIdentifier);
}

void WorkletGlobalScope::addMessage(MessageSource source, MessageLevel level, const String& messageText, const String& sourceURL, unsigned lineNumber, unsigned columnNumber, RefPtr<ScriptCallStack>&& callStack, JSC::JSGlobalObject*, unsigned long requestIdentifier)
{
    if (!m_document || isJSExecutionForbidden())
        return;
    m_document->addMessage(source, level, messageText, sourceURL, lineNumber, columnNumber, WTFMove(callStack), nullptr, requestIdentifier);
}

void WorkletGlobalScope::fetchAndInvokeScript(const URL& moduleURL, FetchRequestCredentials credentials, CompletionHandler<void(std::optional<Exception>&&)>&& completionHandler)
{
    ASSERT(!isMainThread());
    script()->loadAndEvaluateModule(moduleURL, credentials, WTFMove(completionHandler));
}

MessagePortChannelProvider& WorkletGlobalScope::messagePortChannelProvider()
{
    if (!m_messagePortChannelProvider)
        m_messagePortChannelProvider = makeUnique<WorkerMessagePortChannelProvider>(*this);
    return *m_messagePortChannelProvider;
}

} // namespace WebCore

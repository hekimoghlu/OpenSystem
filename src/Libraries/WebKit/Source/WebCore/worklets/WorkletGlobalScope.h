/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 25, 2021.
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
#pragma once

#include "Document.h"
#include "ExceptionOr.h"
#include "FetchRequestCredentials.h"
#include "ScriptExecutionContext.h"
#include "ScriptSourceCode.h"
#include "WorkerOrWorkletGlobalScope.h"
#include "WorkerOrWorkletScriptController.h"
#include <JavaScriptCore/ConsoleMessage.h>
#include <JavaScriptCore/RuntimeFlags.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Deque.h>
#include <wtf/URL.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class MessagePortChannelProvider;
class WorkerMessagePortChannelProvider;
class WorkerScriptLoader;

struct WorkletParameters;

enum class WorkletGlobalScopeIdentifierType { };
using WorkletGlobalScopeIdentifier = ObjectIdentifier<WorkletGlobalScopeIdentifierType>;

class WorkletGlobalScope : public WorkerOrWorkletGlobalScope {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WorkletGlobalScope);
public:
    virtual ~WorkletGlobalScope();

    virtual bool isPaintWorkletGlobalScope() const { return false; }
#if ENABLE(WEB_AUDIO)
    virtual bool isAudioWorkletGlobalScope() const { return false; }
#endif

    WEBCORE_EXPORT static unsigned numberOfWorkletGlobalScopes();

    MessagePortChannelProvider& messagePortChannelProvider();

    const URL& url() const final { return m_url; }
    const URL& cookieURL() const final { return url(); }

    void evaluate();

    void addConsoleMessage(std::unique_ptr<Inspector::ConsoleMessage>&&) final;

    SecurityOrigin& topOrigin() const final { return m_topOrigin.get(); }

    SocketProvider* socketProvider() final { return nullptr; }

    bool isSecureContext() const final { return false; }

    JSC::RuntimeFlags jsRuntimeFlags() const { return m_jsRuntimeFlags; }

    void prepareForDestruction() override;

    void fetchAndInvokeScript(const URL&, FetchRequestCredentials, CompletionHandler<void(std::optional<Exception>&&)>&&);

    Document* responsibleDocument() { return m_document.get(); }
    const Document* responsibleDocument() const { return m_document.get(); }

protected:
    WorkletGlobalScope(WorkerOrWorkletThread&, Ref<JSC::VM>&&, const WorkletParameters&);
    WorkletGlobalScope(Document&, Ref<JSC::VM>&&, ScriptSourceCode&&);

private:
    IDBClient::IDBConnectionProxy* idbConnectionProxy() final { ASSERT_NOT_REACHED(); return nullptr; }

    // EventTarget.
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::WorkletGlobalScope; }

    bool isWorkletGlobalScope() const final { return true; }

    void logExceptionToConsole(const String& errorMessage, const String&, int, int, RefPtr<Inspector::ScriptCallStack>&&) final;
    void addMessage(MessageSource, MessageLevel, const String&, const String&, unsigned, unsigned, RefPtr<Inspector::ScriptCallStack>&&, JSC::JSGlobalObject*, unsigned long) final;
    void addConsoleMessage(MessageSource, MessageLevel, const String&, unsigned long) final;

    EventTarget* errorEventTarget() final { return this; }

    std::optional<Vector<uint8_t>> wrapCryptoKey(const Vector<uint8_t>&) final { RELEASE_ASSERT_NOT_REACHED(); return std::nullopt; }
    std::optional<Vector<uint8_t>> serializeAndWrapCryptoKey(CryptoKeyData&&) final { RELEASE_ASSERT_NOT_REACHED(); return std::nullopt; }
    std::optional<Vector<uint8_t>> unwrapCryptoKey(const Vector<uint8_t>&) final { RELEASE_ASSERT_NOT_REACHED(); return std::nullopt; }
    URL completeURL(const String&, ForceUTF8 = ForceUTF8::No) const final;
    String userAgent(const URL&) const final;
    const Settings::Values& settingsValues() const final { return m_settingsValues; }

    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_document;

    Ref<SecurityOrigin> m_topOrigin;

    URL m_url;
    JSC::RuntimeFlags m_jsRuntimeFlags;
    std::optional<ScriptSourceCode> m_code;

    std::unique_ptr<WorkerMessagePortChannelProvider> m_messagePortChannelProvider;

    Settings::Values m_settingsValues;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::WorkletGlobalScope)
static bool isType(const WebCore::ScriptExecutionContext& context) { return context.isWorkletGlobalScope(); }
SPECIALIZE_TYPE_TRAITS_END()

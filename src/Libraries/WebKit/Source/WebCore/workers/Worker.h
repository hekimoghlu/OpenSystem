/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 14, 2022.
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

#include "AbstractWorker.h"
#include "ActiveDOMObject.h"
#include "ContentSecurityPolicyResponseHeaders.h"
#include "EventTarget.h"
#include "FetchRequestCredentials.h"
#include "MessagePort.h"
#include "WorkerOptions.h"
#include "WorkerScriptLoaderClient.h"
#include "WorkerType.h"
#include <JavaScriptCore/RuntimeFlags.h>
#include <wtf/Deque.h>
#include <wtf/MonotonicTime.h>
#include <wtf/text/AtomStringHash.h>

namespace JSC {
class CallFrame;
class JSObject;
class JSValue;
}

namespace WebCore {

class RTCRtpScriptTransform;
class RTCRtpScriptTransformer;
class ScriptExecutionContext;
class TrustedScriptURL;
class WorkerGlobalScopeProxy;
class WorkerScriptLoader;

struct StructuredSerializeOptions;
struct WorkerOptions;

class Worker final : public AbstractWorker, public ActiveDOMObject, private WorkerScriptLoaderClient {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(Worker);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    USING_CAN_MAKE_WEAKPTR(AbstractWorker);

    static ExceptionOr<Ref<Worker>> create(ScriptExecutionContext&, JSC::RuntimeFlags, std::variant<RefPtr<TrustedScriptURL>, String>&&, WorkerOptions&&);
    virtual ~Worker();

    ExceptionOr<void> postMessage(JSC::JSGlobalObject&, JSC::JSValue message, StructuredSerializeOptions&&);

    void terminate();
    bool wasTerminated() const { return m_wasTerminated; }

    String identifier() const { return m_identifier; }
    const String& name() const { return m_options.name; }

    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }

    void dispatchEvent(Event&) final;
    void reportError(const String&);

#if ENABLE(WEB_RTC)
    void createRTCRtpScriptTransformer(RTCRtpScriptTransform&, MessageWithMessagePorts&&);
#endif

    WorkerType type() const { return m_options.type; }

    void postTaskToWorkerGlobalScope(Function<void(ScriptExecutionContext&)>&&);

    static void forEachWorker(const Function<Function<void(ScriptExecutionContext&)>()>&);

private:
    Worker(ScriptExecutionContext&, JSC::RuntimeFlags, WorkerOptions&&);

    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::Worker; }

    void didReceiveResponse(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const ResourceResponse&) final;
    void notifyFinished(std::optional<ScriptExecutionContextIdentifier>) final;

    // ActiveDOMObject.
    void stop() final;
    void suspend(ReasonForSuspension) final;
    void resume() final;
    bool virtualHasPendingActivity() const final;

    static void networkStateChanged(bool isOnLine);

    RefPtr<WorkerScriptLoader> m_scriptLoader;
    const WorkerOptions m_options;
    String m_identifier;
    WorkerGlobalScopeProxy& m_contextProxy; // The proxy outlives the worker to perform thread shutdown.
    std::optional<ContentSecurityPolicyResponseHeaders> m_contentSecurityPolicyResponseHeaders;
    MonotonicTime m_workerCreationTime;
    bool m_shouldBypassMainWorldContentSecurityPolicy { false };
    bool m_isSuspendedForBackForwardCache { false };
    JSC::RuntimeFlags m_runtimeFlags;
    Deque<RefPtr<Event>> m_pendingEvents;
    bool m_wasTerminated { false };
    bool m_didStartWorkerGlobalScope { false };
    const ScriptExecutionContextIdentifier m_clientIdentifier;
};

} // namespace WebCore

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 8, 2021.
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

#include "MessagePort.h"
#include "WorkerGlobalScope.h"

namespace JSC {
class CallFrame;
class JSObject;
class JSValue;
}

namespace WebCore {

class ContentSecurityPolicyResponseHeaders;
class DedicatedWorkerThread;
class JSRTCRtpScriptTransformerConstructor;
class RTCRtpScriptTransformer;
class RequestAnimationFrameCallback;
class SerializedScriptValue;
struct StructuredSerializeOptions;

#if ENABLE(NOTIFICATIONS)
class WorkerNotificationClient;
#endif

#if ENABLE(OFFSCREEN_CANVAS_IN_WORKERS)
class WorkerAnimationController;

using CallbackId = int;
#endif

using TransferredMessagePort = std::pair<WebCore::MessagePortIdentifier, WebCore::MessagePortIdentifier>;

class DedicatedWorkerGlobalScope final : public WorkerGlobalScope {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DedicatedWorkerGlobalScope);
public:
    static Ref<DedicatedWorkerGlobalScope> create(const WorkerParameters&, Ref<SecurityOrigin>&&, DedicatedWorkerThread&, Ref<SecurityOrigin>&& topOrigin, IDBClient::IDBConnectionProxy*, SocketProvider*, std::unique_ptr<WorkerClient>&&);
    virtual ~DedicatedWorkerGlobalScope();

    const String& name() const { return m_name; }

    ExceptionOr<void> postMessage(JSC::JSGlobalObject&, JSC::JSValue message, StructuredSerializeOptions&&);

    DedicatedWorkerThread& thread();

#if ENABLE(NOTIFICATIONS)
    NotificationClient* notificationClient() final;
#endif

#if ENABLE(OFFSCREEN_CANVAS_IN_WORKERS)
    CallbackId requestAnimationFrame(Ref<RequestAnimationFrameCallback>&&);
    void cancelAnimationFrame(CallbackId);
#endif

#if ENABLE(WEB_RTC)
    RefPtr<RTCRtpScriptTransformer> createRTCRtpScriptTransformer(MessageWithMessagePorts&&);
#endif

    FetchOptions::Destination destination() const final { return FetchOptions::Destination::Worker; }

private:
    using Base = WorkerGlobalScope;

    DedicatedWorkerGlobalScope(const WorkerParameters&, Ref<SecurityOrigin>&&, DedicatedWorkerThread&, Ref<SecurityOrigin>&& topOrigin, IDBClient::IDBConnectionProxy*, SocketProvider*, std::unique_ptr<WorkerClient>&&);

    Type type() const final { return Type::DedicatedWorker; }

    enum EventTargetInterfaceType eventTargetInterface() const final;

    void prepareForDestruction() final;

    String m_name;

#if ENABLE(OFFSCREEN_CANVAS_IN_WORKERS)
    RefPtr<WorkerAnimationController> m_workerAnimationController;
#endif
#if ENABLE(NOTIFICATIONS)
    RefPtr<WorkerNotificationClient> m_notificationClient;
#endif
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::DedicatedWorkerGlobalScope)
    static bool isType(const WebCore::ScriptExecutionContext& context)
    {
        auto* global = dynamicDowncast<WebCore::WorkerGlobalScope>(context);
        return global && global->type() == WebCore::WorkerGlobalScope::Type::DedicatedWorker;
    }
    static bool isType(const WebCore::WorkerGlobalScope& context) { return context.type() == WebCore::WorkerGlobalScope::Type::DedicatedWorker; }
SPECIALIZE_TYPE_TRAITS_END()

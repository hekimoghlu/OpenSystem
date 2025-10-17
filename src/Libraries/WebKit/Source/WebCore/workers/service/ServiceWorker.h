/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 14, 2022.
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

#include "ActiveDOMObject.h"
#include "ContextDestructionObserver.h"
#include "EventTarget.h"
#include "ServiceWorkerData.h"
#include <JavaScriptCore/Strong.h>
#include <wtf/RefCounted.h>
#include <wtf/URL.h>

namespace JSC {
class JSGlobalObject;
class JSValue;
}

namespace WebCore {

class LocalFrame;
class SWClientConnection;

struct StructuredSerializeOptions;

class ServiceWorker final : public RefCounted<ServiceWorker>, public EventTarget, public ActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(ServiceWorker, WEBCORE_EXPORT);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    using State = ServiceWorkerState;
    static Ref<ServiceWorker> getOrCreate(ScriptExecutionContext&, ServiceWorkerData&&);

    WEBCORE_EXPORT virtual ~ServiceWorker();

    const URL& scriptURL() const { return m_data.scriptURL; }

    State state() const { return m_data.state; }
    
    void updateState(State);

    ExceptionOr<void> postMessage(JSC::JSGlobalObject&, JSC::JSValue message, StructuredSerializeOptions&&);

    ServiceWorkerIdentifier identifier() const { return m_data.identifier; }
    ServiceWorkerRegistrationIdentifier registrationIdentifier() const { return m_data.registrationIdentifier; }
    WorkerType workerType() const { return m_data.type; }

    const ServiceWorkerData& data() const { return m_data; }

private:
    ServiceWorker(ScriptExecutionContext&, ServiceWorkerData&&);
    void updatePendingActivityForEventDispatch();

    enum EventTargetInterfaceType eventTargetInterface() const final;
    ScriptExecutionContext* scriptExecutionContext() const final;
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    // ActiveDOMObject.
    void stop() final;

    SWClientConnection& swConnection();

    ServiceWorkerData m_data;
    bool m_isStopped { false };
    RefPtr<PendingActivity<ServiceWorker>> m_pendingActivityForEventDispatch;
};

} // namespace WebCore

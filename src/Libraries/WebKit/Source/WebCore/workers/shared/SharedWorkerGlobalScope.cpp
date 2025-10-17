/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 4, 2023.
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
#include "SharedWorkerGlobalScope.h"

#include "EventNames.h"
#include "Logging.h"
#include "MessageEvent.h"
#include "SerializedScriptValue.h"
#include "ServiceWorkerThread.h"
#include "SharedWorkerThread.h"
#include "WorkerThread.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SharedWorkerGlobalScope);

#define SCOPE_RELEASE_LOG(fmt, ...) RELEASE_LOG(SharedWorker, "%p - [sharedWorkerIdentifier=%" PRIu64 "] SharedWorkerGlobalScope::" fmt, this, this->thread().identifier().toUInt64(), ##__VA_ARGS__)

SharedWorkerGlobalScope::SharedWorkerGlobalScope(const String& name, const WorkerParameters& params, Ref<SecurityOrigin>&& origin, SharedWorkerThread& thread, Ref<SecurityOrigin>&& topOrigin, IDBClient::IDBConnectionProxy* connectionProxy, SocketProvider* socketProvider, std::unique_ptr<WorkerClient>&& workerClient)
    : WorkerGlobalScope(WorkerThreadType::SharedWorker, params, WTFMove(origin), thread, WTFMove(topOrigin), connectionProxy, socketProvider, WTFMove(workerClient))
    , m_name(name)
{
    SCOPE_RELEASE_LOG("SharedWorkerGlobalScope:");
    applyContentSecurityPolicyResponseHeaders(params.contentSecurityPolicyResponseHeaders);
}

SharedWorkerGlobalScope::~SharedWorkerGlobalScope()
{
    // We need to remove from the contexts map very early in the destructor so that calling postTask() on this WorkerGlobalScope from another thread is safe.
    removeFromContextsMap();
}

SharedWorkerThread& SharedWorkerGlobalScope::thread()
{
    return static_cast<SharedWorkerThread&>(WorkerGlobalScope::thread());
}

// https://html.spec.whatwg.org/multipage/workers.html#dom-sharedworker step 11.5
void SharedWorkerGlobalScope::postConnectEvent(TransferredMessagePort&& transferredPort, const String& sourceOrigin)
{
    SCOPE_RELEASE_LOG("postConnectEvent:");
    auto ports = MessagePort::entanglePorts(*this, { WTFMove(transferredPort) });
    ASSERT(ports.size() == 1);
    RefPtr port = ports[0].ptr();
    auto event = MessageEvent::create(emptyString(), sourceOrigin, { }, port, WTFMove(ports));
    event->initEvent(eventNames().connectEvent, false, false);

    dispatchEvent(WTFMove(event));
}

#undef SCOPE_RELEASE_LOG

} // namespace WebCore

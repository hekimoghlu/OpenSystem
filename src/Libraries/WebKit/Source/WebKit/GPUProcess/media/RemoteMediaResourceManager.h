/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 26, 2022.
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

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

#include "Connection.h"
#include "RemoteMediaResourceIdentifier.h"
#include "WorkQueueMessageReceiver.h"
#include <WebCore/PolicyChecker.h>
#include <WebCore/SharedMemory.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace IPC {
class Connection;
class Decoder;
class SharedBufferReference;
}

namespace WebCore {
class NetworkLoadMetrics;
class ResourceRequest;
}

namespace WebKit {

class RemoteMediaResource;

class RemoteMediaResourceManager
    : public IPC::WorkQueueMessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteMediaResourceManager);
public:
    static Ref<RemoteMediaResourceManager> create() { return adoptRef(*new RemoteMediaResourceManager()); }
    ~RemoteMediaResourceManager();

    void ref() const final { IPC::WorkQueueMessageReceiver::ref(); }
    void deref() const final { IPC::WorkQueueMessageReceiver::deref(); }

    void initializeConnection(IPC::Connection*);
    void stopListeningForIPC();

    void addMediaResource(RemoteMediaResourceIdentifier, RemoteMediaResource&);
    void removeMediaResource(RemoteMediaResourceIdentifier);

    // IPC::Connection::WorkQueueMessageReceiver.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

private:
    RemoteMediaResourceManager();
    void responseReceived(RemoteMediaResourceIdentifier, const WebCore::ResourceResponse&, bool, CompletionHandler<void(WebCore::ShouldContinuePolicyCheck)>&&);
    void redirectReceived(RemoteMediaResourceIdentifier, WebCore::ResourceRequest&&, const WebCore::ResourceResponse&, CompletionHandler<void(WebCore::ResourceRequest&&)>&&);
    void dataSent(RemoteMediaResourceIdentifier, uint64_t, uint64_t);
    void dataReceived(RemoteMediaResourceIdentifier, IPC::SharedBufferReference&&, CompletionHandler<void(std::optional<WebCore::SharedMemory::Handle>&&)>&&);
    void accessControlCheckFailed(RemoteMediaResourceIdentifier, const WebCore::ResourceError&);
    void loadFailed(RemoteMediaResourceIdentifier, const WebCore::ResourceError&);
    void loadFinished(RemoteMediaResourceIdentifier, const WebCore::NetworkLoadMetrics&);

    RefPtr<RemoteMediaResource> resourceForId(RemoteMediaResourceIdentifier);

    Lock m_lock;
    HashMap<RemoteMediaResourceIdentifier, ThreadSafeWeakPtr<RemoteMediaResource>> m_remoteMediaResources WTF_GUARDED_BY_LOCK(m_lock);

    RefPtr<IPC::Connection> m_connection;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

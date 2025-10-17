/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 23, 2025.
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

#include "RemoteMediaResourceIdentifier.h"
#include <WebCore/PlatformMediaResourceLoader.h>
#include <atomic>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class NetworkLoadMetrics;
class FragmentedSharedBuffer;
class SharedBuffer;
}

namespace WebKit {

class RemoteMediaPlayerProxy;
class RemoteMediaResourceManager;

class RemoteMediaResource : public WebCore::PlatformMediaResource {
    WTF_MAKE_TZONE_ALLOCATED(RemoteMediaResource);
public:
    // Called on the main thread.
    static Ref<RemoteMediaResource> create(RemoteMediaResourceManager&, RemoteMediaPlayerProxy&, RemoteMediaResourceIdentifier);

    // Thread-safe
    ~RemoteMediaResource();
    void shutdown() final;

    // PlatformMediaResource, called on the main thread.
    bool didPassAccessControlCheck() const final;

    // Called on MediaResourceLoader's WorkQueue.
    void responseReceived(const WebCore::ResourceResponse&, bool, CompletionHandler<void(WebCore::ShouldContinuePolicyCheck)>&&);
    void redirectReceived(WebCore::ResourceRequest&&, const WebCore::ResourceResponse&, CompletionHandler<void(WebCore::ResourceRequest&&)>&&);
    void dataSent(uint64_t, uint64_t);
    void dataReceived(const WebCore::SharedBuffer&);
    void accessControlCheckFailed(const WebCore::ResourceError&);
    void loadFailed(const WebCore::ResourceError&);
    void loadFinished(const WebCore::NetworkLoadMetrics&);

private:
    RemoteMediaResource(RemoteMediaResourceManager&, RemoteMediaPlayerProxy&, RemoteMediaResourceIdentifier);

    ThreadSafeWeakPtr<RemoteMediaResourceManager> m_remoteMediaResourceManager;
    WeakPtr<RemoteMediaPlayerProxy> m_remoteMediaPlayerProxy;
    RemoteMediaResourceIdentifier m_id;
    std::atomic<bool> m_didPassAccessControlCheck { false };
    std::atomic<bool> m_shutdown { false };
};

} // namespace WebKit


#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

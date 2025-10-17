/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 19, 2024.
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
#include <WebCore/PlatformMediaResourceLoader.h>
#include <WebCore/PolicyChecker.h>
#include <WebCore/ResourceResponse.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class RemoteMediaResourceProxy final : public WebCore::PlatformMediaResourceClient {
    WTF_MAKE_TZONE_ALLOCATED(RemoteMediaResourceProxy);
public:
    RemoteMediaResourceProxy(Ref<IPC::Connection>&&, WebCore::PlatformMediaResource&, RemoteMediaResourceIdentifier);
    ~RemoteMediaResourceProxy();

private:
    // PlatformMediaResourceClient
    void responseReceived(WebCore::PlatformMediaResource&, const WebCore::ResourceResponse&, CompletionHandler<void(WebCore::ShouldContinuePolicyCheck)>&&) final;
    void redirectReceived(WebCore::PlatformMediaResource&, WebCore::ResourceRequest&&, const WebCore::ResourceResponse&, CompletionHandler<void(WebCore::ResourceRequest&&)>&&) final;
    bool shouldCacheResponse(WebCore::PlatformMediaResource&, const WebCore::ResourceResponse&) final;
    void dataSent(WebCore::PlatformMediaResource&, unsigned long long, unsigned long long) final;
    void dataReceived(WebCore::PlatformMediaResource&, const WebCore::SharedBuffer&) final;
    void accessControlCheckFailed(WebCore::PlatformMediaResource&, const WebCore::ResourceError&) final;
    void loadFailed(WebCore::PlatformMediaResource&, const WebCore::ResourceError&) final;
    void loadFinished(WebCore::PlatformMediaResource&, const WebCore::NetworkLoadMetrics&) final;

    Ref<WebCore::PlatformMediaResource> protectedMediaResource() const;
    Ref<IPC::Connection> protectedConnection() const { return m_connection; }

    Ref<IPC::Connection> m_connection;
    ThreadSafeWeakPtr<WebCore::PlatformMediaResource> m_platformMediaResource; // Cannot be null.
    RemoteMediaResourceIdentifier m_id;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

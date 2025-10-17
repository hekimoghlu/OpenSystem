/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 10, 2022.
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

#include "ResourceLoader.h"
#include <wtf/Function.h>
#include <wtf/Forward.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class NetscapePlugInStreamLoaderClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::NetscapePlugInStreamLoaderClient> : std::true_type { };
}

namespace WebCore {

class NetscapePlugInStreamLoader;
class SharedBuffer;

class NetscapePlugInStreamLoaderClient : public CanMakeWeakPtr<NetscapePlugInStreamLoaderClient> {
public:
    virtual void willSendRequest(NetscapePlugInStreamLoader*, ResourceRequest&&, const ResourceResponse& redirectResponse, CompletionHandler<void(ResourceRequest&&)>&&) = 0;
    virtual void didReceiveResponse(NetscapePlugInStreamLoader*, const ResourceResponse&) = 0;
    virtual void didReceiveData(NetscapePlugInStreamLoader*, const SharedBuffer&) = 0;
    virtual void didFail(NetscapePlugInStreamLoader*, const ResourceError&) = 0;
    virtual void didFinishLoading(NetscapePlugInStreamLoader*) { }
    virtual bool wantsAllStreams() const { return false; }

protected:
    virtual ~NetscapePlugInStreamLoaderClient() = default;
};

class NetscapePlugInStreamLoader final : public ResourceLoader {
public:
    WEBCORE_EXPORT static void create(LocalFrame&, NetscapePlugInStreamLoaderClient&, ResourceRequest&&, CompletionHandler<void(RefPtr<NetscapePlugInStreamLoader>&&)>&&);
    virtual ~NetscapePlugInStreamLoader();

    WEBCORE_EXPORT bool isDone() const;

private:
    void init(ResourceRequest&&, CompletionHandler<void(bool)>&&) override;

    void willSendRequest(ResourceRequest&&, const ResourceResponse& redirectResponse, CompletionHandler<void(ResourceRequest&&)>&& callback) override;
    void didReceiveResponse(const ResourceResponse&, CompletionHandler<void()>&& policyCompletionHandler) override;
    void didReceiveData(const SharedBuffer&, long long encodedDataLength, DataPayloadType) override;
    void didFinishLoading(const NetworkLoadMetrics&) override;
    void didFail(const ResourceError&) override;

    void releaseResources() override;

    NetscapePlugInStreamLoader(LocalFrame&, NetscapePlugInStreamLoaderClient&);

    void willCancel(const ResourceError&) override;
    void didCancel(LoadWillContinueInAnotherProcess) override;

    void notifyDone();

    WeakPtr<NetscapePlugInStreamLoaderClient> m_client;
    bool m_isInitialized { false };
};

}

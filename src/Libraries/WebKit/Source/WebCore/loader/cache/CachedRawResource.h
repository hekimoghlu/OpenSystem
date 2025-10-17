/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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

#include "CachedResource.h"

namespace WebCore {

class CachedResourceClient;
class ResourceTiming;
class SharedBuffer;
class SharedBufferDataView;

class CachedRawResource final : public CachedResource {
public:
    CachedRawResource(CachedResourceRequest&&, Type, PAL::SessionID, const CookieJar*);

    void setDefersLoading(bool);

    void setDataBufferingPolicy(DataBufferingPolicy);

    // FIXME: This is exposed for the InspectorInstrumentation for preflights in DocumentThreadableLoader. It's also really lame.
    std::optional<ResourceLoaderIdentifier> resourceLoaderIdentifier() const { return m_resourceLoaderIdentifier; }

    void clear();

    bool canReuse(const ResourceRequest&) const;

    bool wasRedirected() const { return !m_redirectChain.isEmpty(); };

    void finishedTimingForWorkerLoad(ResourceTiming&&);

private:
    void didAddClient(CachedResourceClient&) final;
    void updateBuffer(const FragmentedSharedBuffer&) final;
    void updateData(const SharedBuffer&) final;
    void finishLoading(const FragmentedSharedBuffer*, const NetworkLoadMetrics&) final;

    bool shouldIgnoreHTTPStatusCodeErrors() const override { return true; }
    void allClientsRemoved() override;

    void redirectReceived(ResourceRequest&&, const ResourceResponse&, CompletionHandler<void(ResourceRequest&&)>&&) override;
    void responseReceived(const ResourceResponse&) override;
    bool shouldCacheResponse(const ResourceResponse&) override;
    void didSendData(unsigned long long bytesSent, unsigned long long totalBytesToBeSent) override;

    void switchClientsToRevalidatedResource() override;
    bool mayTryReplaceEncodedData() const override { return m_allowEncodedDataReplacement; }

    std::optional<SharedBufferDataView> calculateIncrementalDataChunk(const FragmentedSharedBuffer&) const;
    void notifyClientsDataWasReceived(const SharedBuffer&);
    
#if USE(QUICK_LOOK)
    void previewResponseReceived(const ResourceResponse&) final;
#endif

    Markable<ResourceLoaderIdentifier> m_resourceLoaderIdentifier;

    struct RedirectPair {
    public:
        explicit RedirectPair(const ResourceRequest& request, const ResourceResponse& redirectResponse)
            : m_request(request)
            , m_redirectResponse(redirectResponse)
        {
        }

        const ResourceRequest m_request;
        const ResourceResponse m_redirectResponse;
    };

    Vector<RedirectPair, 0, CrashOnOverflow, 0> m_redirectChain;

    struct DelayedFinishLoading {
        RefPtr<const FragmentedSharedBuffer> buffer;
    };
    std::optional<DelayedFinishLoading> m_delayedFinishLoading;

    bool m_allowEncodedDataReplacement { true };
    bool m_inIncrementalDataNotify { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::CachedRawResource)
    static bool isType(const WebCore::CachedResource& resource) { return resource.isMainOrMediaOrIconOrRawResource(); }
SPECIALIZE_TYPE_TRAITS_END()

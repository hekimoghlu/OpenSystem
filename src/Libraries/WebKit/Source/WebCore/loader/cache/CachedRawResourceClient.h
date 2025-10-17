/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 29, 2024.
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

#include "CachedResourceClient.h"
#include <wtf/CompletionHandler.h>

namespace WebCore {
class CachedRawResourceClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::CachedRawResourceClient> : std::true_type { };
}

namespace WebCore {

class CachedResource;
class ResourceRequest;
class ResourceResponse;
class ResourceTiming;
class SharedBuffer;

class CachedRawResourceClient : public CachedResourceClient {
public:
    static CachedResourceClientType expectedType() { return RawResourceType; }
    CachedResourceClientType resourceClientType() const override { return expectedType(); }

    virtual void dataSent(CachedResource&, unsigned long long /* bytesSent */, unsigned long long /* totalBytesToBeSent */) { }
    virtual void responseReceived(CachedResource&, const ResourceResponse&, CompletionHandler<void()>&& completionHandler)
    {
        if (completionHandler)
            completionHandler();
    }

    virtual bool shouldCacheResponse(CachedResource&, const ResourceResponse&) { return true; }
    virtual void dataReceived(CachedResource&, const SharedBuffer&) { }
    virtual void redirectReceived(CachedResource&, ResourceRequest&& request, const ResourceResponse&, CompletionHandler<void(ResourceRequest&&)>&& completionHandler) { completionHandler(WTFMove(request)); }
    virtual void finishedTimingForWorkerLoad(CachedResource&, const ResourceTiming&) { }

#if USE(QUICK_LOOK)
    virtual void previewResponseReceived(CachedResource&, const ResourceResponse&) { };
#endif
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CACHED_RESOURCE_CLIENT(CachedRawResourceClient, RawResourceType);

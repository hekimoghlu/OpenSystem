/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 13, 2023.
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

#include "ApplicationCacheResource.h"
#include "CachedRawResource.h"
#include "CachedRawResourceClient.h"
#include "CachedResourceHandle.h"
#include <wtf/CompletionHandler.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class CachedResourceLoader;
class ResourceRequest;

class ApplicationCacheResourceLoader final : public RefCounted<ApplicationCacheResourceLoader>, private CachedRawResourceClient {
public:
    enum class Error { Abort, NetworkError, CannotCreateResource, NotFound, NotOK, RedirectForbidden };
    using ResourceOrError = Expected<RefPtr<ApplicationCacheResource>, Error>;

    static RefPtr<ApplicationCacheResourceLoader> create(unsigned, CachedResourceLoader&, ResourceRequest&&, CompletionHandler<void(ResourceOrError&&)>&&);
    ~ApplicationCacheResourceLoader();

    void cancel(Error = Error::Abort);

    const CachedResource* resource() const { return m_resource.get(); }
    bool hasRedirection() const { return m_hasRedirection; }
    unsigned type() const { return m_type; }

private:
    explicit ApplicationCacheResourceLoader(unsigned, CachedResourceHandle<CachedRawResource>&&, CompletionHandler<void(ResourceOrError&&)>&&);

    // CachedRawResourceClient
    void responseReceived(CachedResource&, const ResourceResponse&, CompletionHandler<void()>&&) final;
    void dataReceived(CachedResource&, const SharedBuffer&) final;
    void redirectReceived(CachedResource&, ResourceRequest&&, const ResourceResponse&, CompletionHandler<void(ResourceRequest&&)>&&) final;
    void notifyFinished(CachedResource&, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess) final;

    unsigned m_type;
    CachedResourceHandle<CachedRawResource> m_resource;
    RefPtr<ApplicationCacheResource> m_applicationCacheResource;
    CompletionHandler<void(ResourceOrError&&)> m_callback;
    bool m_hasRedirection { false };
};

} // namespace WebCore

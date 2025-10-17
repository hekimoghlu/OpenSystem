/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 26, 2023.
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

#include "FetchOptions.h"
#include "LoadSchedulingMode.h"
#include "PageIdentifier.h"
#include "ResourceLoadPriority.h"
#include "ResourceLoaderIdentifier.h"
#include "ResourceLoaderOptions.h"
#include "StoredCredentialsPolicy.h"
#include <wtf/Forward.h>

namespace WebCore {

class CachedResource;
class ContentSecurityPolicy;
class FrameLoader;
class HTTPHeaderMap;
class LocalFrame;
class NetscapePlugInStreamLoader;
class NetscapePlugInStreamLoaderClient;
struct NetworkTransactionInformation;
class NetworkLoadMetrics;
class Page;
class ResourceError;
class ResourceLoader;
class ResourceRequest;
class ResourceResponse;
class SecurityOrigin;
class FragmentedSharedBuffer;
class SubresourceLoader;

struct FetchOptions;

class WEBCORE_EXPORT LoaderStrategy {
public:
    virtual void loadResource(LocalFrame&, CachedResource&, ResourceRequest&&, const ResourceLoaderOptions&, CompletionHandler<void(RefPtr<SubresourceLoader>&&)>&&) = 0;
    virtual void loadResourceSynchronously(FrameLoader&, ResourceLoaderIdentifier, const ResourceRequest&, ClientCredentialPolicy, const FetchOptions&, const HTTPHeaderMap&, ResourceError&, ResourceResponse&, Vector<uint8_t>& data) = 0;
    virtual void pageLoadCompleted(Page&) = 0;
    virtual void browsingContextRemoved(LocalFrame&) = 0;

    virtual void remove(ResourceLoader*) = 0;
    virtual void setDefersLoading(ResourceLoader&, bool) = 0;
    virtual void crossOriginRedirectReceived(ResourceLoader*, const URL& redirectURL) = 0;

    virtual void servePendingRequests(ResourceLoadPriority minimumPriority = ResourceLoadPriority::VeryLow) = 0;
    virtual void suspendPendingRequests() = 0;
    virtual void resumePendingRequests() = 0;

    virtual void setResourceLoadSchedulingMode(Page&, LoadSchedulingMode);
    virtual void prioritizeResourceLoads(const Vector<SubresourceLoader*>&);

    virtual bool usePingLoad() const { return true; }
    using PingLoadCompletionHandler = Function<void(const ResourceError&, const ResourceResponse&)>;
    virtual void startPingLoad(LocalFrame&, ResourceRequest&, const HTTPHeaderMap& originalRequestHeaders, const FetchOptions&, ContentSecurityPolicyImposition, PingLoadCompletionHandler&& = { }) = 0;

    using PreconnectCompletionHandler = Function<void(const ResourceError&)>;
    enum class ShouldPreconnectAsFirstParty : bool { No, Yes };
    virtual void preconnectTo(FrameLoader&, ResourceRequest&&, StoredCredentialsPolicy, ShouldPreconnectAsFirstParty, PreconnectCompletionHandler&&) = 0;

    virtual void setCaptureExtraNetworkLoadMetricsEnabled(bool) = 0;

    virtual bool isOnLine() const = 0;
    virtual void addOnlineStateChangeListener(Function<void(bool)>&&) = 0;

    virtual bool shouldPerformSecurityChecks() const { return false; }
    virtual bool havePerformedSecurityChecks(const ResourceResponse&) const { return false; }

    virtual ResourceResponse responseFromResourceLoadIdentifier(ResourceLoaderIdentifier);
    virtual Vector<NetworkTransactionInformation> intermediateLoadInformationFromResourceLoadIdentifier(ResourceLoaderIdentifier);
    virtual NetworkLoadMetrics networkMetricsFromResourceLoadIdentifier(ResourceLoaderIdentifier);

    virtual void isResourceLoadFinished(CachedResource&, CompletionHandler<void(bool)>&& callback) = 0;

    // Used for testing only.
    virtual Vector<ResourceLoaderIdentifier> ongoingLoads() const { return { }; }

    virtual ResourceError cancelledError(const ResourceRequest&) const = 0;
    virtual ResourceError blockedError(const ResourceRequest&) const = 0;
    virtual ResourceError blockedByContentBlockerError(const ResourceRequest&) const = 0;
    virtual ResourceError cannotShowURLError(const ResourceRequest&) const = 0;
    virtual ResourceError interruptedForPolicyChangeError(const ResourceRequest&) const = 0;
#if ENABLE(CONTENT_FILTERING)
    virtual ResourceError blockedByContentFilterError(const ResourceRequest&) const = 0;
#endif
    virtual ResourceError cannotShowMIMETypeError(const ResourceResponse&) const = 0;
    virtual ResourceError fileDoesNotExistError(const ResourceResponse&) const = 0;
    virtual ResourceError httpsUpgradeRedirectLoopError(const ResourceRequest&) const = 0;
    virtual ResourceError httpNavigationWithHTTPSOnlyError(const ResourceRequest&) const = 0;
    virtual ResourceError pluginWillHandleLoadError(const ResourceResponse&) const = 0;

protected:
    virtual ~LoaderStrategy();
};

} // namespace WebCore

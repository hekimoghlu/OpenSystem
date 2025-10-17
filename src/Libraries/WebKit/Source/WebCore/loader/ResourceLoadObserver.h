/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 12, 2024.
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

#include "LoaderMalloc.h"
#include "ResourceLoadStatistics.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>

namespace WebCore {

class Document;
class LocalFrame;
class ResourceRequest;
class ResourceResponse;

class ResourceLoadObserver {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
public:
    using TopFrameDomain = WebCore::RegistrableDomain;
    using SubResourceDomain = WebCore::RegistrableDomain;

    // https://fetch.spec.whatwg.org/#request-destination-script-like
    enum class FetchDestinationIsScriptLike : bool { No, Yes };

    WEBCORE_EXPORT static ResourceLoadObserver& shared();
    WEBCORE_EXPORT static ResourceLoadObserver* sharedIfExists();
    WEBCORE_EXPORT static void setShared(ResourceLoadObserver&);
    
    virtual ~ResourceLoadObserver() { }

    virtual void logSubresourceLoading(const LocalFrame*, const ResourceRequest& /* newRequest */, const ResourceResponse& /* redirectResponse */, FetchDestinationIsScriptLike) { }
    virtual void logWebSocketLoading(const URL& /* targetURL */, const URL& /* mainFrameURL */) { }
    virtual void logUserInteractionWithReducedTimeResolution(const Document&) { }
    virtual void logFontLoad(const Document&, const String& /* familyName */, bool /* loadStatus */) { }
    virtual void logCanvasRead(const Document&) { }
    virtual void logCanvasWriteOrMeasure(const Document&, const String& /* textWritten */) { }
    virtual void logNavigatorAPIAccessed(const Document&, const NavigatorAPIsAccessed) { }
    virtual void logScreenAPIAccessed(const Document&, const ScreenAPIsAccessed) { }
    virtual void logSubresourceLoadingForTesting(const RegistrableDomain& /* firstPartyDomain */, const RegistrableDomain& /* thirdPartyDomain */, bool /* shouldScheduleNotification */) { }

    virtual String statisticsForURL(const URL&) { return { }; }
    virtual void updateCentralStatisticsStore(CompletionHandler<void()>&& completionHandler) { completionHandler(); }
    virtual void clearState() { }
    
    virtual bool hasStatistics() const { return false; }

    virtual void setDomainsWithUserInteraction(HashSet<RegistrableDomain>&&) { }
    virtual void setDomainsWithCrossPageStorageAccess(HashMap<TopFrameDomain, Vector<SubResourceDomain>>&&, CompletionHandler<void()>&& completionHandler) { completionHandler(); }
    virtual bool hasCrossPageStorageAccess(const SubResourceDomain&, const TopFrameDomain&) const { return false; }
    virtual bool hasHadUserInteraction(const RegistrableDomain&) const { return false; }
};
    
} // namespace WebCore

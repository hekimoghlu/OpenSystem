/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 26, 2024.
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

#if ENABLE(CONTENT_FILTERING)

#include "CachedResourceHandle.h"
#include "LoaderMalloc.h"
#include "PlatformContentFilter.h"
#include "ResourceError.h"
#include <functional>
#include <wtf/Forward.h>
#include <wtf/UniqueRef.h>

namespace WebCore {

class CachedRawResource;
class ContentFilterClient;
class DocumentLoader;
class ResourceRequest;
class ResourceResponse;
class SubstituteData;

class ContentFilter {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
    WTF_MAKE_NONCOPYABLE(ContentFilter);

public:
    template <typename T> static void addType() { types().append(type<T>()); }

    WEBCORE_EXPORT static std::unique_ptr<ContentFilter> create(ContentFilterClient&);
    WEBCORE_EXPORT ~ContentFilter();

    static constexpr ASCIILiteral urlScheme() { return "x-apple-content-filter"_s; }

    WEBCORE_EXPORT void startFilteringMainResource(const URL&);
    void startFilteringMainResource(CachedRawResource&);
    WEBCORE_EXPORT void stopFilteringMainResource();

    WEBCORE_EXPORT bool continueAfterWillSendRequest(ResourceRequest&, const ResourceResponse&);
    WEBCORE_EXPORT bool continueAfterResponseReceived(const ResourceResponse&);
    WEBCORE_EXPORT bool continueAfterDataReceived(const SharedBuffer&, size_t encodedDataLength);
    WEBCORE_EXPORT bool continueAfterNotifyFinished(const URL& resourceURL);
    bool continueAfterDataReceived(const SharedBuffer&);
    bool continueAfterNotifyFinished(CachedResource&);

    static bool continueAfterSubstituteDataRequest(const DocumentLoader& activeLoader, const SubstituteData&);
    bool willHandleProvisionalLoadFailure(const ResourceError&) const;
    WEBCORE_EXPORT void handleProvisionalLoadFailure(const ResourceError&);

    const ResourceError& blockedError() const { return m_blockedError; }
    void setBlockedError(const ResourceError& error) { m_blockedError = error; }
    bool isAllowed() const { return m_state == State::Allowed; }
    bool responseReceived() const { return m_responseReceived; }

    WEBCORE_EXPORT static const URL& blockedPageURL();

#if HAVE(AUDIT_TOKEN)
    WEBCORE_EXPORT void setHostProcessAuditToken(const std::optional<audit_token_t>&);
#endif

private:
    using State = PlatformContentFilter::State;

    struct Type {
        Function<UniqueRef<PlatformContentFilter>()> create;
    };
    template <typename T> static Type type();
    WEBCORE_EXPORT static Vector<Type>& types();

    using Container = Vector<UniqueRef<PlatformContentFilter>>;
    friend std::unique_ptr<ContentFilter> std::make_unique<ContentFilter>(Container&&, ContentFilterClient&);
    ContentFilter(Container&&, ContentFilterClient&);

    template <typename Function> void forEachContentFilterUntilBlocked(Function&&);
    void didDecide(State);
    void deliverResourceData(const SharedBuffer&, size_t encodedDataLength = 0);
    void deliverStoredResourceData();

    Ref<ContentFilterClient> protectedClient() const;
    
    URL url();
    
    Container m_contentFilters;
    WeakRef<ContentFilterClient> m_client;
    URL m_mainResourceURL;
    struct ResourceDataItem {
        RefPtr<const SharedBuffer> buffer;
        size_t encodedDataLength;
    };
    Vector<ResourceDataItem> m_buffers;
    CachedResourceHandle<CachedRawResource> m_mainResource;
    WeakPtr<const PlatformContentFilter> m_blockingContentFilter;
    State m_state { State::Stopped };
    ResourceError m_blockedError;
    bool m_isLoadingBlockedPage { false };
    bool m_responseReceived { false };
};

template <typename T>
ContentFilter::Type ContentFilter::type()
{
    static_assert(std::is_base_of<PlatformContentFilter, T>::value, "Type must be a PlatformContentFilter.");
    return { T::create };
}

} // namespace WebCore

#endif // ENABLE(CONTENT_FILTERING)

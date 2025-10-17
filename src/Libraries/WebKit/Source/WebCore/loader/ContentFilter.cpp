/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 1, 2024.
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
#include "config.h"
#include "ContentFilter.h"

#if ENABLE(CONTENT_FILTERING)

#include "CachedRawResource.h"
#include "ContentFilterClient.h"
#include "ContentFilterUnblockHandler.h"
#include "DocumentLoader.h"
#include "FrameLoadRequest.h"
#include "FrameLoader.h"
#include "LocalFrame.h"
#include "LocalFrameLoaderClient.h"
#include "Logging.h"
#include "NetworkExtensionContentFilter.h"
#include "ParentalControlsContentFilter.h"
#include "ScriptController.h"
#include "SharedBuffer.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/Ref.h>
#include <wtf/SetForScope.h>
#include <wtf/Vector.h>

#if !LOG_DISABLED
#include <wtf/text/CString.h>
#endif

namespace WebCore {

Vector<ContentFilter::Type>& ContentFilter::types()
{
    static NeverDestroyed<Vector<ContentFilter::Type>> types {
        Vector<ContentFilter::Type>::from(
#if HAVE(PARENTAL_CONTROLS)
            type<ParentalControlsContentFilter>(),
#endif
            type<NetworkExtensionContentFilter>()
        )
    };
    return types;
}

std::unique_ptr<ContentFilter> ContentFilter::create(ContentFilterClient& client)
{
    auto filters = types().map([](auto& type) {
        return type.create();
    });

    if (filters.isEmpty())
        return nullptr;

    return makeUnique<ContentFilter>(WTFMove(filters), client);
}

ContentFilter::ContentFilter(Container&& contentFilters, ContentFilterClient& client)
    : m_contentFilters(WTFMove(contentFilters))
    , m_client(client)
{
    LOG(ContentFiltering, "Creating ContentFilter with %zu platform content filter(s).\n", m_contentFilters.size());
    ASSERT(!m_contentFilters.isEmpty());
}

ContentFilter::~ContentFilter()
{
    LOG(ContentFiltering, "Destroying ContentFilter.\n");
}

bool ContentFilter::continueAfterWillSendRequest(ResourceRequest& request, const ResourceResponse& redirectResponse)
{
    Ref protectedClient { m_client.get() };

    LOG(ContentFiltering, "ContentFilter received request for <%{sensitive}s> with redirect response from <%{sensitive}s>.\n", request.url().string().ascii().data(), redirectResponse.url().string().ascii().data());
#if !LOG_DISABLED
    ResourceRequest originalRequest { request };
#endif
    ASSERT(m_state == State::Stopped || m_state == State::Filtering);
    forEachContentFilterUntilBlocked([&request, &redirectResponse](PlatformContentFilter& contentFilter) {
        contentFilter.willSendRequest(request, redirectResponse);
    });
    if (m_state == State::Blocked)
        request = ResourceRequest();
#if !LOG_DISABLED
    if (request != originalRequest)
        LOG(ContentFiltering, "ContentFilter changed request url to <%{sensitive}s>.\n", originalRequest.url().string().ascii().data());
#endif
    return !request.isNull();
}

void ContentFilter::startFilteringMainResource(const URL& url)
{
    if (m_state != State::Stopped)
        return;

    LOG(ContentFiltering, "ContentFilter will start filtering main resource at <%{sensitive}s>.\n", url.string().ascii().data());
    m_state = State::Filtering;
    ASSERT(m_mainResourceURL.isEmpty());
    m_mainResourceURL = url;
}

void ContentFilter::startFilteringMainResource(CachedRawResource& resource)
{
    if (m_state != State::Stopped)
        return;

    LOG(ContentFiltering, "ContentFilter will start filtering main resource at <%{sensitive}s>.\n", resource.url().string().ascii().data());
    m_state = State::Filtering;
    ASSERT(!m_mainResource);
    m_mainResource = resource;
}

void ContentFilter::stopFilteringMainResource()
{
    if (m_state != State::Blocked)
        m_state = State::Stopped;
    m_mainResourceURL = URL();
}

bool ContentFilter::continueAfterResponseReceived(const ResourceResponse& response)
{
    Ref protectedClient { m_client.get() };

    if (m_state == State::Filtering) {
        LOG(ContentFiltering, "ContentFilter received response from <%{sensitive}s>.\n", response.url().string().ascii().data());
        forEachContentFilterUntilBlocked([&response](PlatformContentFilter& contentFilter) {
            contentFilter.responseReceived(response);
        });
    }

    m_responseReceived = true;

    return m_state != State::Blocked;
}

bool ContentFilter::continueAfterDataReceived(const SharedBuffer& data, size_t encodedDataLength)
{
    Ref protectedClient { m_client.get() };

    if (m_state == State::Filtering) {
        LOG(ContentFiltering, "ContentFilter received %zu bytes of data from <%{sensitive}s>.\n", data.size(), url().string().ascii().data());

        forEachContentFilterUntilBlocked([data = Ref { data }](auto& contentFilter) {
            contentFilter.addData(data);
        });
        if (m_state == State::Allowed) {
            deliverStoredResourceData();
            deliverResourceData(data, encodedDataLength);
        } else
            m_buffers.append(ResourceDataItem { RefPtr { &data }, encodedDataLength });
        return false;
    }

    return m_state != State::Blocked;
}

bool ContentFilter::continueAfterDataReceived(const SharedBuffer& data)
{
    Ref protectedClient { m_client.get() };

    if (m_state == State::Filtering) {
        LOG(ContentFiltering, "ContentFilter received %zu bytes of data from <%{sensitive}s>.\n", data.size(), url().string().ascii().data());

        forEachContentFilterUntilBlocked([data = Ref { data }](auto& contentFilter) {
            contentFilter.addData(data);
        });
        if (m_state == State::Allowed) {
            ASSERT(m_mainResource->dataBufferingPolicy() == DataBufferingPolicy::BufferData);
            if (RefPtr buffer = m_mainResource->resourceBuffer())
                deliverResourceData(buffer->makeContiguous());
        }
        return false;
    }

    return m_state != State::Blocked;
}

bool ContentFilter::continueAfterNotifyFinished(const URL& resourceURL)
{
    Ref protectedClient { m_client.get() };
    ASSERT_UNUSED(resourceURL, resourceURL == m_mainResourceURL);

    if (m_state == State::Filtering) {
        LOG(ContentFiltering, "ContentFilter will finish filtering main resource at <%{sensitive}s>.\n", url().string().ascii().data());
        forEachContentFilterUntilBlocked([](PlatformContentFilter& contentFilter) {
            contentFilter.finishedAddingData();
        });

        if (m_state != State::Blocked) {
            m_state = State::Allowed;
            deliverStoredResourceData();
        }

        if (m_state == State::Stopped)
            return false;
    }

    return m_state != State::Blocked;
}

bool ContentFilter::continueAfterNotifyFinished(CachedResource& resource)
{
    Ref protectedClient { m_client.get() };
    ASSERT_UNUSED(resource, &resource == m_mainResource);
    if (m_mainResource->errorOccurred())
        return true;

    if (m_state == State::Filtering) {
        LOG(ContentFiltering, "ContentFilter will finish filtering main resource at <%{sensitive}s>.\n", url().string().ascii().data());
        forEachContentFilterUntilBlocked([](PlatformContentFilter& contentFilter) {
            contentFilter.finishedAddingData();
        });

        if (m_state != State::Blocked) {
            m_state = State::Allowed;
            if (RefPtr buffer = m_mainResource->resourceBuffer()) {
                ASSERT(m_mainResource->dataBufferingPolicy() == DataBufferingPolicy::BufferData);
                deliverResourceData(buffer->makeContiguous());
            }
        }

        if (m_state == State::Stopped)
            return false;
    }

    return m_state != State::Blocked;
}

template <typename Function>
inline void ContentFilter::forEachContentFilterUntilBlocked(Function&& function)
{
    bool allFiltersAllowedLoad { true };
    for (auto& contentFilter : m_contentFilters) {
        if (!contentFilter->needsMoreData()) {
            ASSERT(!contentFilter->didBlockData());
            continue;
        }

        function(contentFilter.get());

        if (contentFilter->didBlockData()) {
            ASSERT(!m_blockingContentFilter);
            m_blockingContentFilter = contentFilter.get();
            didDecide(State::Blocked);
            return;
        } else if (contentFilter->needsMoreData())
            allFiltersAllowedLoad = false;
    }

    if (allFiltersAllowedLoad)
        didDecide(State::Allowed);
}

void ContentFilter::didDecide(State state)
{
    ASSERT(m_state != State::Allowed);
    ASSERT(m_state != State::Blocked);
    ASSERT(state == State::Allowed || state == State::Blocked);
#if !LOG_DISABLED
    LOG(ContentFiltering, "ContentFilter decided load should be %s for main resource at <%{sensitive}s>.\n", state == State::Allowed ? "allowed" : "blocked", url().string().ascii().data());
#endif // !LOG_DISABLED
    m_state = state;
    if (m_state != State::Blocked)
        return;

    Ref client = m_client.get();
    m_blockedError = client->contentFilterDidBlock(m_blockingContentFilter->unblockHandler(), m_blockingContentFilter->unblockRequestDeniedScript());
    client->cancelMainResourceLoadForContentFilter(m_blockedError);
}

Ref<ContentFilterClient> ContentFilter::protectedClient() const
{
    return m_client.get();
}

void ContentFilter::deliverResourceData(const SharedBuffer& buffer, size_t encodedDataLength)
{
    ASSERT(m_state == State::Allowed);
    protectedClient()->dataReceivedThroughContentFilter(buffer, encodedDataLength);
}

URL ContentFilter::url()
{
    if (m_mainResource)
        return m_mainResource->url();
    return m_mainResourceURL;
}

const URL& ContentFilter::blockedPageURL()
{
    static NeverDestroyed blockedPageURL = [] () -> URL {
        auto webCoreBundle = CFBundleGetBundleWithIdentifier(CFSTR("com.apple.WebCore"));
        return adoptCF(CFBundleCopyResourceURL(webCoreBundle, CFSTR("ContentFilterBlockedPage"), CFSTR("html"), nullptr)).get();
    }();
    return blockedPageURL;
}

bool ContentFilter::continueAfterSubstituteDataRequest(const DocumentLoader& activeLoader, const SubstituteData& substituteData)
{
    if (auto contentFilter = activeLoader.contentFilter()) {
        if (contentFilter->m_state == State::Blocked && !contentFilter->m_isLoadingBlockedPage)
            return contentFilter->m_blockedError.failingURL() != substituteData.failingURL();
    }

    if (activeLoader.request().url() == blockedPageURL()) {
        ASSERT(activeLoader.substituteData().isValid());
        return activeLoader.substituteData().failingURL() != substituteData.failingURL();
    }

    return true;
}

bool ContentFilter::willHandleProvisionalLoadFailure(const ResourceError& error) const
{
    if (m_state != State::Blocked)
        return false;

    if (m_blockedError.errorCode() != error.errorCode() || m_blockedError.domain() != error.domain())
        return false;

    ASSERT(m_blockedError.failingURL() == error.failingURL());
    return true;
}

void ContentFilter::handleProvisionalLoadFailure(const ResourceError& error)
{
    ASSERT(willHandleProvisionalLoadFailure(error));

    RefPtr replacementData { m_blockingContentFilter->replacementData() };
    ResourceResponse response { URL(), "text/html"_s, static_cast<long long>(replacementData->size()), "UTF-8"_s };
    SubstituteData substituteData { WTFMove(replacementData), error.failingURL(), response, SubstituteData::SessionHistoryVisibility::Hidden };
    SetForScope loadingBlockedPage { m_isLoadingBlockedPage, true };
    protectedClient()->handleProvisionalLoadFailureFromContentFilter(blockedPageURL(), substituteData);
}

void ContentFilter::deliverStoredResourceData()
{
    for (auto& buffer : m_buffers)
        deliverResourceData(*buffer.buffer, buffer.encodedDataLength);
    m_buffers.clear();
}

#if HAVE(AUDIT_TOKEN)
void ContentFilter::setHostProcessAuditToken(const std::optional<audit_token_t>& token)
{
    for (auto& contentFilter : m_contentFilters)
        contentFilter->setHostProcessAuditToken(token);
}
#endif

} // namespace WebCore

#endif // ENABLE(CONTENT_FILTERING)

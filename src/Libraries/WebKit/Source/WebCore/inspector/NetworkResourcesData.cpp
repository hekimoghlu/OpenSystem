/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 17, 2024.
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
#include "NetworkResourcesData.h"

#include "CachedResource.h"
#include "InspectorNetworkAgent.h"
#include "ResourceResponse.h"
#include "TextResourceDecoder.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/Base64.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(NetworkResourcesData);
WTF_MAKE_TZONE_ALLOCATED_IMPL(NetworkResourcesData::ResourceData);

using namespace Inspector;

static const size_t maximumResourcesContentSize = 200 * 1000 * 1000; // 200MB
static const size_t maximumSingleResourceContentSize = 50 * 1000 * 1000; // 50MB

NetworkResourcesData::ResourceData::ResourceData(const String& requestId, const String& loaderId)
    : m_requestId(requestId)
    , m_loaderId(loaderId)
{
}

void NetworkResourcesData::ResourceData::setContent(const String& content, bool base64Encoded)
{
    ASSERT(!hasData());
    ASSERT(!hasContent());
    m_content = content;
    m_base64Encoded = base64Encoded;
}

unsigned NetworkResourcesData::ResourceData::removeContent()
{
    unsigned result = 0;
    if (hasData()) {
        ASSERT(!hasContent());
        result = m_dataBuffer.size();
        m_dataBuffer.reset();
    }

    if (hasContent()) {
        ASSERT(!hasData());
        result = m_content.sizeInBytes();
        m_content = String();
    }
    return result;
}

unsigned NetworkResourcesData::ResourceData::evictContent()
{
    m_isContentEvicted = true;
    return removeContent();
}

bool NetworkResourcesData::ResourceData::hasData() const
{
    return !!m_dataBuffer;
}

size_t NetworkResourcesData::ResourceData::dataLength() const
{
    return m_dataBuffer.size();
}

void NetworkResourcesData::ResourceData::appendData(const SharedBuffer& data)
{
    ASSERT(!hasContent());
    m_dataBuffer.append(data);
}

void NetworkResourcesData::ResourceData::decodeDataToContent()
{
    ASSERT(!hasContent());

    auto buffer = m_dataBuffer.takeAsContiguous();

    if (m_decoder) {
        m_base64Encoded = false;
        m_content = m_decoder->decodeAndFlush(buffer->span());
    } else {
        m_base64Encoded = true;
        m_content = base64EncodeToString(buffer->span());
    }
}

NetworkResourcesData::NetworkResourcesData()
    : m_maximumResourcesContentSize(maximumResourcesContentSize)
    , m_maximumSingleResourceContentSize(maximumSingleResourceContentSize)
{
}

NetworkResourcesData::~NetworkResourcesData()
{
    clear();
}

void NetworkResourcesData::resourceCreated(const String& requestId, const String& loaderId, InspectorPageAgent::ResourceType type)
{
    ensureNoDataForRequestId(requestId);

    auto resourceData = makeUnique<ResourceData>(requestId, loaderId);
    resourceData->setType(type);
    m_requestIdToResourceDataMap.set(requestId, WTFMove(resourceData));
}

void NetworkResourcesData::resourceCreated(const String& requestId, const String& loaderId, CachedResource& cachedResource)
{
    ensureNoDataForRequestId(requestId);

    auto resourceData = makeUnique<ResourceData>(requestId, loaderId);
    resourceData->setCachedResource(&cachedResource);
    m_requestIdToResourceDataMap.set(requestId, WTFMove(resourceData));
}

void NetworkResourcesData::responseReceived(const String& requestId, const String& frameId, const ResourceResponse& response, InspectorPageAgent::ResourceType type, bool forceBufferData)
{
    ResourceData* resourceData = resourceDataForRequestId(requestId);
    if (!resourceData)
        return;

    resourceData->setFrameId(frameId);
    resourceData->setURL(response.url().string());
    resourceData->setHTTPStatusCode(response.httpStatusCode());
    resourceData->setHTTPStatusText(response.httpStatusText());
    resourceData->setType(type);
    resourceData->setForceBufferData(forceBufferData);
    resourceData->setMIMEType(response.mimeType());
    resourceData->setResponseTimestamp(WallTime::now());

    if (InspectorNetworkAgent::shouldTreatAsText(response.mimeType()))
        resourceData->setDecoder(InspectorNetworkAgent::createTextDecoder(response.mimeType(), response.textEncodingName()));

    if (auto& certificateInfo = response.certificateInfo())
        resourceData->setCertificateInfo(certificateInfo);
}

void NetworkResourcesData::setResourceType(const String& requestId, InspectorPageAgent::ResourceType type)
{
    ResourceData* resourceData = resourceDataForRequestId(requestId);
    if (!resourceData)
        return;
    resourceData->setType(type);
}

InspectorPageAgent::ResourceType NetworkResourcesData::resourceType(const String& requestId)
{
    ResourceData* resourceData = resourceDataForRequestId(requestId);
    if (!resourceData)
        return InspectorPageAgent::OtherResource;
    return resourceData->type();
}

void NetworkResourcesData::setResourceContent(const String& requestId, const String& content, bool base64Encoded)
{
    if (content.isNull())
        return;

    ResourceData* resourceData = resourceDataForRequestId(requestId);
    if (!resourceData)
        return;

    size_t dataLength = content.sizeInBytes();
    if (dataLength > m_maximumSingleResourceContentSize)
        return;
    if (resourceData->isContentEvicted())
        return;

    if (ensureFreeSpace(dataLength) && !resourceData->isContentEvicted()) {
        // We can not be sure that we didn't try to save this request data while it was loading, so remove it, if any.
        if (resourceData->hasContent() || resourceData->hasData())
            m_contentSize -= resourceData->removeContent();
        m_requestIdsDeque.appendOrMoveToLast(requestId);
        resourceData->setContent(content, base64Encoded);
        m_contentSize += dataLength;
    }
}

static bool shouldBufferResourceData(const NetworkResourcesData::ResourceData& resourceData)
{
    if (resourceData.forceBufferData())
        return true;

    if (resourceData.decoder())
        return true;

    // Buffer data for Web Inspector when the rest of the system would not normally buffer.
    if (resourceData.cachedResource() && resourceData.cachedResource()->dataBufferingPolicy() == DataBufferingPolicy::DoNotBufferData)
        return true;

    return false;
}

NetworkResourcesData::ResourceData const* NetworkResourcesData::maybeAddResourceData(const String& requestId, const SharedBuffer& data)
{
    ResourceData* resourceData = resourceDataForRequestId(requestId);
    if (!resourceData)
        return nullptr;

    if (!shouldBufferResourceData(*resourceData))
        return resourceData;

    if (resourceData->dataLength() + data.size() > m_maximumSingleResourceContentSize)
        m_contentSize -= resourceData->evictContent();
    if (resourceData->isContentEvicted())
        return resourceData;

    if (ensureFreeSpace(data.size()) && !resourceData->isContentEvicted()) {
        m_requestIdsDeque.appendOrMoveToLast(requestId);
        resourceData->appendData(data);
        m_contentSize += data.size();
    }

    return resourceData;
}

void NetworkResourcesData::maybeDecodeDataToContent(const String& requestId)
{
    ResourceData* resourceData = resourceDataForRequestId(requestId);
    if (!resourceData)
        return;

    if (!resourceData->hasData())
        return;

    auto byteCount = resourceData->dataLength();
    m_contentSize -= byteCount;

    resourceData->decodeDataToContent();
    byteCount = resourceData->content().sizeInBytes();
    if (byteCount > m_maximumSingleResourceContentSize) {
        resourceData->evictContent();
        return;
    }

    if (ensureFreeSpace(byteCount) && !resourceData->isContentEvicted())
        m_contentSize += byteCount;
}

void NetworkResourcesData::addCachedResource(const String& requestId, CachedResource* cachedResource)
{
    ResourceData* resourceData = resourceDataForRequestId(requestId);
    if (!resourceData)
        return;
    resourceData->setCachedResource(cachedResource);
}

void NetworkResourcesData::addResourceSharedBuffer(const String& requestId, RefPtr<FragmentedSharedBuffer>&& buffer, const String& textEncodingName)
{
    ResourceData* resourceData = resourceDataForRequestId(requestId);
    if (!resourceData)
        return;
    resourceData->setBuffer(WTFMove(buffer));
    resourceData->setTextEncodingName(textEncodingName);
}

NetworkResourcesData::ResourceData const* NetworkResourcesData::data(const String& requestId)
{
    return resourceDataForRequestId(requestId);
}

NetworkResourcesData::ResourceData const* NetworkResourcesData::dataForURL(const String& url)
{
    if (url.isNull())
        return nullptr;
    
    NetworkResourcesData::ResourceData* mostRecentResourceData = nullptr;
    
    for (auto* resourceData : resources()) {
        // responseTimestamp is checked so that we only grab the most recent response for the URL, instead of potentionally getting a more stale response.
        if (resourceData->url() == url && resourceData->httpStatusCode() != 304 && (!mostRecentResourceData || (resourceData->responseTimestamp() > mostRecentResourceData->responseTimestamp())))
            mostRecentResourceData = resourceData;
    }
    
    return mostRecentResourceData;
}

Vector<String> NetworkResourcesData::removeCachedResource(CachedResource* cachedResource)
{
    Vector<String> result;
    for (auto& entry : m_requestIdToResourceDataMap) {
        ResourceData* resourceData = entry.value.get();
        if (resourceData->cachedResource() == cachedResource) {
            resourceData->setCachedResource(nullptr);
            result.append(entry.key);
        }
    }

    return result;
}

void NetworkResourcesData::clear(std::optional<String> preservedLoaderId)
{
    if (!preservedLoaderId) {
        m_requestIdToResourceDataMap.clear();
        m_requestIdsDeque.clear();
        m_contentSize = 0;
        return;
    }

    for (auto&& requestId : std::exchange(m_requestIdsDeque, { })) {
        auto resourceData = resourceDataForRequestId(requestId);
        if (!resourceData)
            continue;
        if (resourceData->loaderId() == *preservedLoaderId)
            m_requestIdsDeque.add(requestId);
        else {
            m_contentSize -= resourceData->evictContent();
            m_requestIdToResourceDataMap.remove(requestId);
        }
    }
}

Vector<NetworkResourcesData::ResourceData*> NetworkResourcesData::resources()
{
    return WTF::map(m_requestIdToResourceDataMap.values(), [] (const auto& v) { return v.get(); });
}

NetworkResourcesData::ResourceData* NetworkResourcesData::resourceDataForRequestId(const String& requestId)
{
    if (requestId.isNull())
        return nullptr;
    return m_requestIdToResourceDataMap.get(requestId);
}

void NetworkResourcesData::ensureNoDataForRequestId(const String& requestId)
{
    auto result = m_requestIdToResourceDataMap.take(requestId);
    if (!result)
        return;

    ResourceData* resourceData = result.get();
    if (resourceData->hasContent() || resourceData->hasData())
        m_contentSize -= resourceData->evictContent();
}

bool NetworkResourcesData::ensureFreeSpace(size_t size)
{
    if (size > m_maximumResourcesContentSize)
        return false;

    ASSERT(m_maximumResourcesContentSize >= m_contentSize);
    while (size > m_maximumResourcesContentSize - m_contentSize) {
        String requestId = m_requestIdsDeque.takeFirst();
        ResourceData* resourceData = resourceDataForRequestId(requestId);
        if (resourceData)
            m_contentSize -= resourceData->evictContent();
    }
    return true;
}

} // namespace WebCore

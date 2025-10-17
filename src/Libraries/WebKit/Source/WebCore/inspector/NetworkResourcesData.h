/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 14, 2021.
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

#include "InspectorPageAgent.h"
#include "SharedBuffer.h"
#include <wtf/ListHashSet.h>
#include <wtf/RobinHoodHashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WallTime.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class CachedResource;
class ResourceResponse;
class TextResourceDecoder;

class NetworkResourcesData {
    WTF_MAKE_TZONE_ALLOCATED(NetworkResourcesData);
public:
    class ResourceData {
        WTF_MAKE_TZONE_ALLOCATED(ResourceData);
        friend class NetworkResourcesData;
    public:
        ResourceData(const String& requestId, const String& loaderId);

        const String& requestId() const { return m_requestId; }
        const String& loaderId() const { return m_loaderId; }

        const String& frameId() const { return m_frameId; }
        void setFrameId(const String& frameId) { m_frameId = frameId; }

        const String& url() const { return m_url; }
        void setURL(const String& url) { m_url = url; }

        bool hasContent() const { return !m_content.isNull(); }
        const String& content() const { return m_content; }
        void setContent(const String&, bool base64Encoded);

        bool base64Encoded() const { return m_base64Encoded; }

        unsigned removeContent();
        unsigned evictContent();
        bool isContentEvicted() const { return m_isContentEvicted; }

        InspectorPageAgent::ResourceType type() const { return m_type; }
        void setType(InspectorPageAgent::ResourceType type) { m_type = type; }

        int httpStatusCode() const { return m_httpStatusCode; }
        void setHTTPStatusCode(int httpStatusCode) { m_httpStatusCode = httpStatusCode; }
        
        const String& httpStatusText() const { return m_httpStatusText; }
        void setHTTPStatusText(const String& httpStatusText) { m_httpStatusText = httpStatusText; }

        const String& textEncodingName() const { return m_textEncodingName; }
        void setTextEncodingName(const String& textEncodingName) { m_textEncodingName = textEncodingName; }
        
        const String& mimeType() const { return m_mimeType; }
        void setMIMEType(const String& mimeType) { m_mimeType = mimeType; }

        RefPtr<TextResourceDecoder> decoder() const { return m_decoder.copyRef(); }
        void setDecoder(RefPtr<TextResourceDecoder>&& decoder) { m_decoder = WTFMove(decoder); }

        RefPtr<FragmentedSharedBuffer> buffer() const { return m_buffer.copyRef(); }
        void setBuffer(RefPtr<FragmentedSharedBuffer>&& buffer) { m_buffer = WTFMove(buffer); }

        const std::optional<CertificateInfo>& certificateInfo() const { return m_certificateInfo; }
        void setCertificateInfo(const std::optional<CertificateInfo>& certificateInfo) { m_certificateInfo = certificateInfo; }

        CachedResource* cachedResource() const { return m_cachedResource; }
        void setCachedResource(CachedResource* cachedResource) { m_cachedResource = cachedResource; }

        bool forceBufferData() const { return m_forceBufferData; }
        void setForceBufferData(bool force) { m_forceBufferData = force; }
        
        WallTime responseTimestamp() const { return m_responseTimestamp; }
        void setResponseTimestamp(WallTime time) { m_responseTimestamp = time; }

        bool hasBufferedData() const { return hasData(); }

    private:
        bool hasData() const;
        size_t dataLength() const;
        void appendData(const SharedBuffer&);
        void decodeDataToContent();

        String m_requestId;
        String m_loaderId;
        String m_frameId;
        String m_url;
        String m_content;
        String m_textEncodingName;
        String m_mimeType;
        RefPtr<TextResourceDecoder> m_decoder;
        SharedBufferBuilder m_dataBuffer;
        RefPtr<FragmentedSharedBuffer> m_buffer;
        std::optional<CertificateInfo> m_certificateInfo;
        CachedResource* m_cachedResource { nullptr };
        InspectorPageAgent::ResourceType m_type { InspectorPageAgent::OtherResource };
        int m_httpStatusCode { 0 };
        String m_httpStatusText;
        bool m_isContentEvicted { false };
        bool m_base64Encoded { false };
        bool m_forceBufferData { false };
        WallTime m_responseTimestamp;
    };

    NetworkResourcesData();
    ~NetworkResourcesData();

    void resourceCreated(const String& requestId, const String& loaderId, InspectorPageAgent::ResourceType);
    void resourceCreated(const String& requestId, const String& loaderId, CachedResource&);
    void responseReceived(const String& requestId, const String& frameId, const ResourceResponse&, InspectorPageAgent::ResourceType, bool forceBufferData);
    void setResourceType(const String& requestId, InspectorPageAgent::ResourceType);
    InspectorPageAgent::ResourceType resourceType(const String& requestId);
    void setResourceContent(const String& requestId, const String& content, bool base64Encoded = false);
    ResourceData const* maybeAddResourceData(const String& requestId, const SharedBuffer&);
    void maybeDecodeDataToContent(const String& requestId);
    void addCachedResource(const String& requestId, CachedResource*);
    void addResourceSharedBuffer(const String& requestId, RefPtr<FragmentedSharedBuffer>&&, const String& textEncodingName);
    ResourceData const* data(const String& requestId);
    ResourceData const* dataForURL(const String& url);
    Vector<String> removeCachedResource(CachedResource*);
    void clear(std::optional<String> preservedLoaderId = std::nullopt);
    Vector<ResourceData*> resources();

private:
    ResourceData* resourceDataForRequestId(const String& requestId);
    void ensureNoDataForRequestId(const String& requestId);
    bool ensureFreeSpace(size_t);

    ListHashSet<String> m_requestIdsDeque;
    MemoryCompactRobinHoodHashMap<String, std::unique_ptr<ResourceData>> m_requestIdToResourceDataMap;
    size_t m_contentSize { 0 };
    size_t m_maximumResourcesContentSize;
    size_t m_maximumSingleResourceContentSize;
};

} // namespace WebCore

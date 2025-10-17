/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 24, 2022.
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

#include "NetworkCacheStorage.h"
#include "PrivateRelayed.h"
#include <WebCore/ResourceRequest.h>
#include <WebCore/ResourceResponse.h>
#include <WebCore/ShareableResource.h>
#include <wtf/Noncopyable.h>
#include <wtf/Seconds.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class FragmentedSharedBuffer;
}

namespace WebKit::NetworkCache {

class Entry {
    WTF_MAKE_TZONE_ALLOCATED(Entry);
public:
    Entry(const Key&, const WebCore::ResourceResponse&, PrivateRelayed, RefPtr<WebCore::FragmentedSharedBuffer>&&, const Vector<std::pair<String, String>>& varyingRequestHeaders);
    Entry(const Key&, const WebCore::ResourceResponse&, const WebCore::ResourceRequest& redirectRequest, const Vector<std::pair<String, String>>& varyingRequestHeaders);
    explicit Entry(const Storage::Record&);
    Entry(const Entry&);

    Storage::Record encodeAsStorageRecord() const;
    static std::unique_ptr<Entry> decodeStorageRecord(const Storage::Record&);

    PrivateRelayed privateRelayed() const { return m_privateRelayed; }
    const Key& key() const { return m_key; }
    WallTime timeStamp() const { return m_timeStamp; }
    const WebCore::ResourceResponse& response() const { return m_response; }
    const Vector<std::pair<String, String>>& varyingRequestHeaders() const { return m_varyingRequestHeaders; }

    WebCore::FragmentedSharedBuffer* buffer() const;
    RefPtr<WebCore::FragmentedSharedBuffer> protectedBuffer() const;
    const std::optional<WebCore::ResourceRequest>& redirectRequest() const { return m_redirectRequest; }

#if ENABLE(SHAREABLE_RESOURCE)
    std::optional<WebCore::ShareableResource::Handle> shareableResourceHandle() const;
#endif

    bool needsValidation() const;
    void setNeedsValidation(bool);

    const Storage::Record& sourceStorageRecord() const { return m_sourceStorageRecord; }

    void asJSON(StringBuilder&, const Storage::RecordInfo&) const;

    bool hasReachedPrevalentResourceAgeCap() const;
    void capMaxAge(const Seconds);

private:
    void initializeBufferFromStorageRecord() const;

    Key m_key;
    WallTime m_timeStamp;
    WebCore::ResourceResponse m_response;
    Vector<std::pair<String, String>> m_varyingRequestHeaders;

    std::optional<WebCore::ResourceRequest> m_redirectRequest;
    mutable RefPtr<WebCore::FragmentedSharedBuffer> m_buffer;
#if ENABLE(SHAREABLE_RESOURCE)
    mutable RefPtr<WebCore::ShareableResource> m_shareableResource;
#endif

    Storage::Record m_sourceStorageRecord { };
    
    std::optional<Seconds> m_maxAgeCap;
    PrivateRelayed m_privateRelayed { PrivateRelayed::No };
};

}

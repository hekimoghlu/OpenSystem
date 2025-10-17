/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 6, 2024.
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

#include "BlobData.h"
#include "BlobRegistry.h"
#include "SecurityOriginData.h"
#include <wtf/HashCountedSet.h>
#include <wtf/RobinHoodHashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/URLHash.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class ResourceHandle;
class ResourceHandleClient;
class ResourceRequest;
class ThreadSafeDataBuffer;
struct PolicyContainer;

// BlobRegistryImpl is not thread-safe. It should only be called from main thread.
class WEBCORE_EXPORT BlobRegistryImpl {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(BlobRegistryImpl, WEBCORE_EXPORT);
public:
    virtual ~BlobRegistryImpl();

    BlobData* getBlobDataFromURL(const URL&, const std::optional<SecurityOriginData>& topOrigin = std::nullopt) const;

    Ref<ResourceHandle> createResourceHandle(const ResourceRequest&, ResourceHandleClient*);

    void appendStorageItems(BlobData*, const BlobDataItemList&, long long offset, long long length);

    void registerInternalFileBlobURL(const URL&, Ref<BlobDataFileReference>&&, const String& contentType);
    void registerInternalBlobURL(const URL&, Vector<BlobPart>&&, const String& contentType);
    void registerBlobURL(const URL&, const URL& srcURL, const PolicyContainer&, const std::optional<SecurityOriginData>& topOrigin);
    void registerInternalBlobURLOptionallyFileBacked(const URL&, const URL& srcURL, RefPtr<BlobDataFileReference>&&, const String& contentType, const PolicyContainer&);
    void registerInternalBlobURLForSlice(const URL&, const URL& srcURL, long long start, long long end, const String& contentType);
    void unregisterBlobURL(const URL&, const std::optional<WebCore::SecurityOriginData>& topOrigin);

    void registerBlobURLHandle(const URL&, const std::optional<WebCore::SecurityOriginData>& topOrigin);
    void unregisterBlobURLHandle(const URL&, const std::optional<WebCore::SecurityOriginData>& topOrigin);

    String blobType(const URL&);
    unsigned long long blobSize(const URL&);

    void writeBlobsToTemporaryFilesForIndexedDB(const Vector<String>& blobURLs, CompletionHandler<void(Vector<String>&& filePaths)>&&);

    struct BlobForFileWriting {
        String blobURL;
        Vector<std::pair<String, RefPtr<DataSegment>>> filePathsOrDataBuffers;
    };

    bool populateBlobsForFileWriting(const Vector<String>& blobURLs, Vector<BlobForFileWriting>&);
    Vector<RefPtr<BlobDataFileReference>> filesInBlob(const URL&, const std::optional<WebCore::SecurityOriginData>& topOrigin = std::nullopt) const;

    void setFileDirectory(String&&);
    void setPartitioningEnabled(bool);

private:
    void registerBlobURLOptionallyFileBacked(const URL&, const URL& srcURL, RefPtr<BlobDataFileReference>&&, const String& contentType, const PolicyContainer&, const std::optional<SecurityOriginData>& topOrigin = std::nullopt);
    void addBlobData(const String& url, RefPtr<BlobData>&&, const std::optional<WebCore::SecurityOriginData>& topOrigin = std::nullopt);
    Ref<DataSegment> createDataSegment(Vector<uint8_t>&&, BlobData&);

    HashCountedSet<String> m_blobReferences;
    MemoryCompactRobinHoodHashMap<String, RefPtr<BlobData>> m_blobs;
    using URLToTopOriginHashMap = MemoryCompactRobinHoodHashMap<String, WebCore::SecurityOriginData>;
    std::optional<URLToTopOriginHashMap> m_allowedBlobURLTopOrigins;
    String m_fileDirectory;
};

inline void BlobRegistryImpl::setFileDirectory(String&& filePath)
{
    m_fileDirectory = WTFMove(filePath);
}

} // namespace WebCore

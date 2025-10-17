/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 16, 2024.
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

#include <optional>
#include <wtf/Forward.h>

namespace WebCore {

class BlobDataFileReference;
class BlobPart;
class BlobRegistry;
class BlobRegistryImpl;
class SecurityOriginData;

struct PolicyContainer;

WEBCORE_EXPORT BlobRegistry& blobRegistry();

// BlobRegistry is not thread-safe. It should only be called from main thread.
class WEBCORE_EXPORT BlobRegistry {
public:

    // Registers a blob URL referring to the specified file.
    virtual void registerInternalFileBlobURL(const URL&, Ref<BlobDataFileReference>&&, const String& path, const String& contentType) = 0;

    // Registers a blob URL referring to the specified blob data.
    virtual void registerInternalBlobURL(const URL&, Vector<BlobPart>&&, const String& contentType) = 0;
    
    // Registers a new blob URL referring to the blob data identified by the specified srcURL.
    virtual void registerBlobURL(const URL&, const URL& srcURL, const PolicyContainer&, const std::optional<SecurityOriginData>& topOrigin) = 0;

    // Registers a new blob URL referring to the blob data identified by the specified srcURL or, if none found, referring to the file found at the given path.
    virtual void registerInternalBlobURLOptionallyFileBacked(const URL&, const URL& srcURL, RefPtr<BlobDataFileReference>&&, const String& contentType) = 0;

    // Negative start and end values select from the end.
    virtual void registerInternalBlobURLForSlice(const URL&, const URL& srcURL, long long start, long long end, const String& contentType) = 0;

    virtual void unregisterBlobURL(const URL&, const std::optional<SecurityOriginData>& topOrigin) = 0;

    virtual void registerBlobURLHandle(const URL&, const std::optional<SecurityOriginData>& topOrigin) = 0;
    virtual void unregisterBlobURLHandle(const URL&, const std::optional<SecurityOriginData>& topOrigin) = 0;

    virtual String blobType(const URL&) = 0;

    virtual unsigned long long blobSize(const URL&) = 0;

    virtual void writeBlobsToTemporaryFilesForIndexedDB(const Vector<String>& blobURLs, CompletionHandler<void(Vector<String>&& filePaths)>&&) = 0;

    virtual BlobRegistryImpl* blobRegistryImpl() { return nullptr; }

protected:
    virtual ~BlobRegistry();
};

} // namespace WebCore

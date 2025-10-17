/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 22, 2021.
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

#include <WebCore/BlobRegistry.h>

namespace WebKit {

class BlobRegistryProxy final : public WebCore::BlobRegistry {
public:
    void registerInternalFileBlobURL(const URL&, Ref<WebCore::BlobDataFileReference>&&, const String& path, const String& contentType) final;
    void registerInternalBlobURL(const URL&, Vector<WebCore::BlobPart>&&, const String& contentType) final;
    void registerBlobURL(const URL&, const URL& srcURL, const WebCore::PolicyContainer&, const std::optional<WebCore::SecurityOriginData>& topOrigin) final;
    void registerInternalBlobURLOptionallyFileBacked(const URL&, const URL& srcURL, RefPtr<WebCore::BlobDataFileReference>&&, const String& contentType) final;
    void unregisterBlobURL(const URL&, const std::optional<WebCore::SecurityOriginData>& topOrigin) final;
    void registerInternalBlobURLForSlice(const URL&, const URL& srcURL, long long start, long long end, const String& contentType) final;
    String blobType(const URL&) final;
    unsigned long long blobSize(const URL&) final;
    void writeBlobsToTemporaryFilesForIndexedDB(const Vector<String>& blobURLs, CompletionHandler<void(Vector<String>&& filePaths)>&&) final;
    void registerBlobURLHandle(const URL&, const std::optional<WebCore::SecurityOriginData>& topOrigin) final;
    void unregisterBlobURLHandle(const URL&, const std::optional<WebCore::SecurityOriginData>& topOrigin) final;
};

}

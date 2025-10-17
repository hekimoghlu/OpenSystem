/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 14, 2023.
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
#include "NetworkProcessPlatformStrategies.h"

#include <WebCore/BlobRegistry.h>
#include <wtf/NeverDestroyed.h>

namespace WebKit {
using namespace WebCore;

void NetworkProcessPlatformStrategies::initialize()
{
    static NeverDestroyed<NetworkProcessPlatformStrategies> platformStrategies;
    setPlatformStrategies(&platformStrategies.get());
}

LoaderStrategy* NetworkProcessPlatformStrategies::createLoaderStrategy()
{
    return nullptr;
}

PasteboardStrategy* NetworkProcessPlatformStrategies::createPasteboardStrategy()
{
    return nullptr;
}

MediaStrategy* NetworkProcessPlatformStrategies::createMediaStrategy()
{
    return nullptr;
}

BlobRegistry* NetworkProcessPlatformStrategies::createBlobRegistry()
{
    using namespace WebCore;
    class EmptyBlobRegistry : public WebCore::BlobRegistry {
        void registerInternalFileBlobURL(const URL&, Ref<BlobDataFileReference>&&, const String& path, const String& contentType) final { ASSERT_NOT_REACHED(); }
        void registerInternalBlobURL(const URL&, Vector<BlobPart>&&, const String& contentType) final { ASSERT_NOT_REACHED(); }
        void registerBlobURL(const URL&, const URL& srcURL, const PolicyContainer&, const std::optional<SecurityOriginData>&) final { ASSERT_NOT_REACHED(); }
        void registerInternalBlobURLOptionallyFileBacked(const URL&, const URL& srcURL, RefPtr<BlobDataFileReference>&&, const String& contentType) final { ASSERT_NOT_REACHED(); }
        void registerInternalBlobURLForSlice(const URL&, const URL& srcURL, long long start, long long end, const String& contentType) final { ASSERT_NOT_REACHED(); }
        void unregisterBlobURL(const URL&, const std::optional<WebCore::SecurityOriginData>&) final { ASSERT_NOT_REACHED(); }
        String blobType(const URL&) final { ASSERT_NOT_REACHED(); return emptyString(); }
        unsigned long long blobSize(const URL&) final { ASSERT_NOT_REACHED(); return 0; }
        void writeBlobsToTemporaryFilesForIndexedDB(const Vector<String>& blobURLs, CompletionHandler<void(Vector<String>&& filePaths)>&&) final { ASSERT_NOT_REACHED(); }
        void registerBlobURLHandle(const URL&, const std::optional<WebCore::SecurityOriginData>&) final { ASSERT_NOT_REACHED(); }
        void unregisterBlobURLHandle(const URL&, const std::optional<WebCore::SecurityOriginData>&) final { ASSERT_NOT_REACHED(); }
    };
    static NeverDestroyed<EmptyBlobRegistry> blobRegistry;
    return &blobRegistry.get();
}

#if ENABLE(DECLARATIVE_WEB_PUSH)
PushStrategy* NetworkProcessPlatformStrategies::createPushStrategy()
{
    return nullptr;
}
#endif

}

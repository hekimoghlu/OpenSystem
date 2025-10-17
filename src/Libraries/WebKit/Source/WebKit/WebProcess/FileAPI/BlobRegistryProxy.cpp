/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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
#include "BlobRegistryProxy.h"

#include "NetworkConnectionToWebProcessMessages.h"
#include "NetworkProcessConnection.h"
#include "WebProcess.h"
#include <WebCore/BlobDataFileReference.h>
#include <WebCore/CrossOriginOpenerPolicy.h>
#include <WebCore/SWContextManager.h>

namespace WebKit {
using namespace WebCore;

void BlobRegistryProxy::registerInternalFileBlobURL(const URL& url, Ref<BlobDataFileReference>&& file, const String& path, const String& contentType)
{
    SandboxExtension::Handle extensionHandle;

    // File path can be empty when submitting a form file input without a file, see bug 111778.
    if (!file->path().isEmpty()) {
        if (auto handle = SandboxExtension::createHandle(file->path(), SandboxExtension::Type::ReadOnly))
            extensionHandle = WTFMove(*handle);
    }

    String replacementPath = path == file->path() ? nullString() : file->path();
    WebProcess::singleton().ensureNetworkProcessConnection().connection().send(Messages::NetworkConnectionToWebProcess::RegisterInternalFileBlobURL(url, path, replacementPath, WTFMove(extensionHandle), contentType), 0);
}

void BlobRegistryProxy::registerInternalBlobURL(const URL& url, Vector<BlobPart>&& blobParts, const String& contentType)
{
    WebProcess::singleton().ensureNetworkProcessConnection().connection().send(Messages::NetworkConnectionToWebProcess::RegisterInternalBlobURL(url, blobParts, contentType), 0);
}

void BlobRegistryProxy::registerBlobURL(const URL& url, const URL& srcURL, const PolicyContainer& policyContainer, const std::optional<SecurityOriginData>& topOrigin)
{
    WebProcess::singleton().ensureNetworkProcessConnection().connection().send(Messages::NetworkConnectionToWebProcess::RegisterBlobURL { url, srcURL, policyContainer, topOrigin }, 0);
}

void BlobRegistryProxy::registerInternalBlobURLOptionallyFileBacked(const URL& url, const URL& srcURL, RefPtr<WebCore::BlobDataFileReference>&& file, const String& contentType)
{
    ASSERT(file);
    WebProcess::singleton().ensureNetworkProcessConnection().connection().send(Messages::NetworkConnectionToWebProcess::RegisterInternalBlobURLOptionallyFileBacked(url, srcURL, file->path(), contentType), 0);
}

void BlobRegistryProxy::unregisterBlobURL(const URL& url, const std::optional<SecurityOriginData>& topOrigin)
{
    WebProcess::singleton().ensureNetworkProcessConnection().connection().send(Messages::NetworkConnectionToWebProcess::UnregisterBlobURL(url, topOrigin), 0);
}

void BlobRegistryProxy::registerInternalBlobURLForSlice(const URL& url, const URL& srcURL, long long start, long long end, const String& contentType)
{
    WebProcess::singleton().ensureNetworkProcessConnection().connection().send(Messages::NetworkConnectionToWebProcess::RegisterInternalBlobURLForSlice(url, srcURL, start, end, contentType), 0);
}

void BlobRegistryProxy::registerBlobURLHandle(const URL& url, const std::optional<SecurityOriginData>& topOrigin)
{
    WebProcess::singleton().ensureNetworkProcessConnection().connection().send(Messages::NetworkConnectionToWebProcess::RegisterBlobURLHandle(url, topOrigin), 0);
}

void BlobRegistryProxy::unregisterBlobURLHandle(const URL& url, const std::optional<SecurityOriginData>& topOrigin)
{
    WebProcess::singleton().ensureNetworkProcessConnection().connection().send(Messages::NetworkConnectionToWebProcess::UnregisterBlobURLHandle(url, topOrigin), 0);
}

String BlobRegistryProxy::blobType(const URL& url)
{
    auto sendResult = WebProcess::singleton().ensureNetworkProcessConnection().connection().sendSync(Messages::NetworkConnectionToWebProcess::BlobType(url), 0);
    auto [result] = sendResult.takeReplyOr(emptyString());
    return result;
}

unsigned long long BlobRegistryProxy::blobSize(const URL& url)
{
    auto sendResult = WebProcess::singleton().ensureNetworkProcessConnection().connection().sendSync(Messages::NetworkConnectionToWebProcess::BlobSize(url), 0);
    auto [resultSize] = sendResult.takeReplyOr(0);
    return resultSize;
}

void BlobRegistryProxy::writeBlobsToTemporaryFilesForIndexedDB(const Vector<String>& blobURLs, CompletionHandler<void(Vector<String>&& filePaths)>&& completionHandler)
{
    WebProcess::singleton().ensureNetworkProcessConnection().writeBlobsToTemporaryFilesForIndexedDB(blobURLs, WTFMove(completionHandler));
}

}

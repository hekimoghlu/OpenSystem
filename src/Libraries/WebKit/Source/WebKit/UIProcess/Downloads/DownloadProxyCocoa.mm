/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 21, 2025.
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
#import "config.h"
#import "DownloadProxy.h"

#import "APIDownloadClient.h"
#import "NetworkProcessMessages.h"
#import "NetworkProcessProxy.h"
#import "WebsiteDataStore.h"

#import <wtf/cocoa/SpanCocoa.h>

#if HAVE(MODERN_DOWNLOADPROGRESS)
#import <WebKitAdditions/DownloadProgressAdditions.h>
#endif

namespace WebKit {

void DownloadProxy::publishProgress(const URL& url)
{
    if (!m_dataStore)
        return;

#if HAVE(MODERN_DOWNLOADPROGRESS)
    RetainPtr localURL = adoptNS([[NSURL alloc] initFileURLWithPath:url.fileSystemPath() relativeToURL:nil]);
    NSError *error = nil;
    RetainPtr bookmark = [localURL bookmarkDataWithOptions:NSURLBookmarkCreationMinimalBookmark includingResourceValuesForKeys:nil relativeToURL:nil error:&error];
    m_dataStore->networkProcess().send(Messages::NetworkProcess::PublishDownloadProgress(m_downloadID, url, span(bookmark.get()), UseDownloadPlaceholder::No, activityAccessToken().span()), 0);
#else
    auto handle = SandboxExtension::createHandle(url.fileSystemPath(), SandboxExtension::Type::ReadWrite);
    ASSERT(handle);
    if (!handle)
        return;

    protectedDataStore()->protectedNetworkProcess()->send(Messages::NetworkProcess::PublishDownloadProgress(m_downloadID, url, WTFMove(*handle)), 0);
#endif
}

#if HAVE(MODERN_DOWNLOADPROGRESS)
void DownloadProxy::didReceivePlaceholderURL(const URL& placeholderURL, std::span<const uint8_t> bookmarkData, WebKit::SandboxExtensionHandle&& handle, CompletionHandler<void()>&& completionHandler)
{
    if (auto placeholderFileExtension = SandboxExtension::create(WTFMove(handle))) {
        bool ok = placeholderFileExtension->consume();
        ASSERT_UNUSED(ok, ok);
    }
    m_client->didReceivePlaceholderURL(*this, placeholderURL, bookmarkData, WTFMove(completionHandler));
}

void DownloadProxy::didReceiveFinalURL(const URL& finalURL, std::span<const uint8_t> bookmarkData, WebKit::SandboxExtensionHandle&& handle)
{
    if (auto completedFileExtension = SandboxExtension::create(WTFMove(handle))) {
        bool ok = completedFileExtension->consume();
        ASSERT_UNUSED(ok, ok);
    }
    m_client->didReceiveFinalURL(*this, finalURL, bookmarkData);
}

void DownloadProxy::didStartUpdatingProgress()
{
    m_assertion = nullptr;
}

Vector<uint8_t> DownloadProxy::bookmarkDataForURL(const URL& url)
{
    RetainPtr localURL = adoptNS([[NSURL alloc] initFileURLWithPath:url.fileSystemPath() relativeToURL:nil]);
    NSError *error = nil;
    RetainPtr bookmark = [localURL bookmarkDataWithOptions:NSURLBookmarkCreationMinimalBookmark includingResourceValuesForKeys:nil relativeToURL:nil error:&error];
    return span(bookmark.get());
}

Vector<uint8_t> DownloadProxy::activityAccessToken()
{
    return ::activityAccessToken();
}

#endif

}

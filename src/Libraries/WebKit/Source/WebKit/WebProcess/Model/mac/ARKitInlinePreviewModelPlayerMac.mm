/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 18, 2024.
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
#import "ARKitInlinePreviewModelPlayerMac.h"

#if ENABLE(ARKIT_INLINE_PREVIEW_MAC)

#import "DrawingArea.h"
#import "Logging.h"
#import "MessageSenderInlines.h"
#import "WebPage.h"
#import "WebPageProxyMessages.h"
#import <WebCore/Model.h>
#import <pal/spi/cocoa/QuartzCoreSPI.h>
#import <pal/spi/mac/SystemPreviewSPI.h>
#import <wtf/MachSendRight.h>
#import <wtf/SoftLinking.h>
#import <wtf/UUID.h>
#import <wtf/text/MakeString.h>

SOFT_LINK_PRIVATE_FRAMEWORK(AssetViewer);
SOFT_LINK_CLASS(AssetViewer, ASVInlinePreview);

namespace WebKit {

Ref<ARKitInlinePreviewModelPlayerMac> ARKitInlinePreviewModelPlayerMac::create(WebPage& page, WebCore::ModelPlayerClient& client)
{
    return adoptRef(*new ARKitInlinePreviewModelPlayerMac(page, client));
}

ARKitInlinePreviewModelPlayerMac::ARKitInlinePreviewModelPlayerMac(WebPage& page, WebCore::ModelPlayerClient& client)
    : ARKitInlinePreviewModelPlayer(page, client)
{
}

ARKitInlinePreviewModelPlayerMac::~ARKitInlinePreviewModelPlayerMac()
{
    if (m_inlinePreview) {
        if (auto* page = this->page())
            page->send(Messages::WebPageProxy::ModelElementDestroyRemotePreview([m_inlinePreview uuid].UUIDString));
    }
    clearFile();
}

std::optional<ModelIdentifier> ARKitInlinePreviewModelPlayerMac::modelIdentifier()
{
    if (m_inlinePreview)
        return { { [m_inlinePreview uuid].UUIDString } };
    return { };
}

static String& sharedModelElementCacheDirectory()
{
    static NeverDestroyed<String> sharedModelElementCacheDirectory;
    return sharedModelElementCacheDirectory;
}

const String& ARKitInlinePreviewModelPlayerMac::modelElementCacheDirectory()
{
    return sharedModelElementCacheDirectory();
}

void ARKitInlinePreviewModelPlayerMac::setModelElementCacheDirectory(const String& path)
{
    sharedModelElementCacheDirectory() = path;
}

void ARKitInlinePreviewModelPlayerMac::createFile(WebCore::Model& modelSource)
{
    // The need for a file is only temporary due to the nature of ASVInlinePreview,
    // https://bugs.webkit.org/show_bug.cgi?id=227567.

    clearFile();

    auto pathToDirectory = modelElementCacheDirectory();
    if (pathToDirectory.isEmpty())
        return;

    auto directoryExists = FileSystem::fileExists(pathToDirectory);
    if (directoryExists && FileSystem::fileTypeFollowingSymlinks(pathToDirectory) != FileSystem::FileType::Directory) {
        ASSERT_NOT_REACHED();
        return;
    }
    if (!directoryExists && !FileSystem::makeAllDirectories(pathToDirectory)) {
        ASSERT_NOT_REACHED();
        return;
    }

    // We need to support .reality files as well, https://bugs.webkit.org/show_bug.cgi?id=227568.
    String fileName = makeString(WTF::UUID::createVersion4(), ".usdz"_s);
    auto filePath = FileSystem::pathByAppendingComponent(pathToDirectory, fileName);
    auto file = FileSystem::openFile(filePath, FileSystem::FileOpenMode::Truncate);
    if (file <= 0)
        return;

    FileSystem::writeToFile(file, modelSource.data()->makeContiguous()->span());
    FileSystem::closeFile(file);
    m_filePath = filePath;
}

void ARKitInlinePreviewModelPlayerMac::clearFile()
{
    if (m_filePath.isEmpty())
        return;

    FileSystem::deleteFile(m_filePath);
    m_filePath = emptyString();
}

// MARK: - WebCore::ModelPlayer overrides.

void ARKitInlinePreviewModelPlayerMac::load(WebCore::Model& modelSource, WebCore::LayoutSize size)
{
    m_size = size;

    auto strongClient = client();
    if (!strongClient)
        return;

    RefPtr strongPage = page();
    if (!strongPage) {
        strongClient->didFailLoading(*this, WebCore::ResourceError { WebCore::errorDomainWebKitInternal, 0, modelSource.url(), "WebPage destroyed"_s });
        return;
    }

    createFile(modelSource);
    createPreviewsForModelWithURL(modelSource.url());
}

void ARKitInlinePreviewModelPlayerMac::createPreviewsForModelWithURL(const URL& url)
{
    // First, create the WebProcess preview.
    m_inlinePreview = adoptNS([allocASVInlinePreviewInstance() initWithFrame:CGRectMake(0, 0, m_size.width(), m_size.height())]);
    LOG(ModelElement, "ARKitInlinePreviewModelPlayerMac::createPreviewsForModelWithURL() created preview with UUID %s and size %f x %f.", ((String)[m_inlinePreview uuid].UUIDString).utf8().data(), m_size.width(), m_size.height());

    auto strongClient = client();
    if (!strongClient)
        return;

    RefPtr strongPage = page();
    if (!strongPage) {
        strongClient->didFailLoading(*this, WebCore::ResourceError { WebCore::errorDomainWebKitInternal, 0, url, "WebPage destroyed"_s });
        return;
    }

    CompletionHandler<void(Expected<std::pair<String, uint32_t>, WebCore::ResourceError>)> completionHandler = [weakSelf = WeakPtr { *this }, url] (Expected<std::pair<String, uint32_t>, WebCore::ResourceError> result) mutable {
        RefPtr strongSelf = weakSelf.get();
        if (!strongSelf)
            return;

        auto strongClient = strongSelf->client();
        if (!strongClient)
            return;

        if (!result) {
            LOG(ModelElement, "ARKitInlinePreviewModelPlayerMac::createPreviewsForModelWithURL() received error from UIProcess");
            strongClient->didFailLoading(*strongSelf, result.error());
            return;
        }

        auto& [uuid, contextId] = *result;
        String expectedUUID = [strongSelf->m_inlinePreview uuid].UUIDString;

        if (uuid != expectedUUID) {
            LOG(ModelElement, "ARKitInlinePreviewModelPlayerMac::createPreviewsForModelWithURL() UUID mismatch, received %s but expected %s.", uuid.utf8().data(), expectedUUID.utf8().data());
            strongClient->didFailLoading(*strongSelf, WebCore::ResourceError { WebCore::errorDomainWebKitInternal, 0, { }, makeString("ARKitInlinePreviewModelPlayer::createPreviewsForModelWithURL() UUID mismatch, received "_s, uuid, " but expected "_s, expectedUUID, '.') });
            return;
        }

        [strongSelf->m_inlinePreview setRemoteContext:contextId];
        LOG(ModelElement, "ARKitInlinePreviewModelPlayerMac::createPreviewsForModelWithURL() successfully established remote connection for UUID %s.", uuid.utf8().data());

        strongSelf->didCreateRemotePreviewForModelWithURL(url);
    };

    // Then, create the UIProcess preview.
    strongPage->sendWithAsyncReply(Messages::WebPageProxy::ModelElementCreateRemotePreview([m_inlinePreview uuid].UUIDString, m_size), WTFMove(completionHandler));
}

void ARKitInlinePreviewModelPlayerMac::didCreateRemotePreviewForModelWithURL(const URL& url)
{
    auto strongClient = client();
    if (!strongClient)
        return;

    RefPtr strongPage = page();
    if (!strongPage) {
        strongClient->didFailLoading(*this, WebCore::ResourceError { WebCore::errorDomainWebKitInternal, 0, url, "WebPage destroyed"_s });
        return;
    }

    CompletionHandler<void(std::optional<WebCore::ResourceError>&&)> completionHandler = [weakSelf = WeakPtr { *this }] (std::optional<WebCore::ResourceError>&& error) mutable {
        RefPtr strongSelf = weakSelf.get();
        if (!strongSelf)
            return;

        auto strongClient = strongSelf->client();
        if (!strongClient)
            return;

        if (error) {
            LOG(ModelElement, "ARKitInlinePreviewModelPlayer::didCreateRemotePreviewForModelWithURL() received error from UIProcess");
            strongClient->didFailLoading(*strongSelf, *error);
            return;
        }

        LOG(ModelElement, "ARKitInlinePreviewModelPlayer::didCreateRemotePreviewForModelWithURL() successfully completed load for UUID %s.", [strongSelf->m_inlinePreview uuid]);

        strongClient->didFinishLoading(*strongSelf);
    };

    // Now that both the WebProcess and UIProcess previews are created, load the file into the remote preview.
    strongPage->sendWithAsyncReply(Messages::WebPageProxy::ModelElementLoadRemotePreview([m_inlinePreview uuid].UUIDString, URL::fileURLWithFileSystemPath(m_filePath)), WTFMove(completionHandler));
}

void ARKitInlinePreviewModelPlayerMac::sizeDidChange(WebCore::LayoutSize size)
{
    if (m_size == size)
        return;

    m_size = size;

    RefPtr strongPage = page();
    if (!strongPage)
        return;

    String uuid = [m_inlinePreview uuid].UUIDString;
    CompletionHandler<void(Expected<MachSendRight, WebCore::ResourceError>)> completionHandler = [weakSelf = WeakPtr { *this }, strongPage, size] (Expected<MachSendRight, WebCore::ResourceError> result) mutable {
        if (!result)
            return;

        RefPtr strongSelf = weakSelf.get();
        if (!strongSelf)
            return;

        auto* drawingArea = strongPage->drawingArea();
        if (!drawingArea)
            return;

        auto fenceSendRight = WTFMove(*result);
        drawingArea->addFence(fenceSendRight);

        [strongSelf->m_inlinePreview setFrameWithinFencedTransaction:CGRectMake(0, 0, size.width(), size.height())];
    };

    strongPage->sendWithAsyncReply(Messages::WebPageProxy::ModelElementSizeDidChange(uuid, size), WTFMove(completionHandler));
}

PlatformLayer* ARKitInlinePreviewModelPlayerMac::layer()
{
    return [m_inlinePreview layer];
}

bool ARKitInlinePreviewModelPlayerMac::supportsMouseInteraction()
{
    return true;
}

bool ARKitInlinePreviewModelPlayerMac::supportsDragging()
{
    return false;
}

void ARKitInlinePreviewModelPlayerMac::handleMouseDown(const LayoutPoint& flippedLocationInElement, MonotonicTime timestamp)
{
    if (auto* page = this->page())
        page->send(Messages::WebPageProxy::HandleMouseDownForModelElement([m_inlinePreview uuid].UUIDString, flippedLocationInElement, timestamp));
}

void ARKitInlinePreviewModelPlayerMac::handleMouseMove(const LayoutPoint& flippedLocationInElement, MonotonicTime timestamp)
{
    if (auto* page = this->page())
        page->send(Messages::WebPageProxy::HandleMouseMoveForModelElement([m_inlinePreview uuid].UUIDString, flippedLocationInElement, timestamp));
}

void ARKitInlinePreviewModelPlayerMac::handleMouseUp(const LayoutPoint& flippedLocationInElement, MonotonicTime timestamp)
{
    if (auto* page = this->page())
        page->send(Messages::WebPageProxy::HandleMouseUpForModelElement([m_inlinePreview uuid].UUIDString, flippedLocationInElement, timestamp));
}

String ARKitInlinePreviewModelPlayerMac::inlinePreviewUUIDForTesting() const
{
    if (!m_inlinePreview)
        return emptyString();
    return [m_inlinePreview uuid].UUIDString;
}

}

#endif

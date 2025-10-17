/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 4, 2024.
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
#import "ARKitInlinePreviewModelPlayer.h"

#if ENABLE(ARKIT_INLINE_PREVIEW)

#import "MessageSenderInlines.h"
#import "WebPage.h"

namespace WebKit {

ARKitInlinePreviewModelPlayer::ARKitInlinePreviewModelPlayer(WebPage& page, WebCore::ModelPlayerClient& client)
    : m_page { page }
    , m_client { client }
{
}

ARKitInlinePreviewModelPlayer::~ARKitInlinePreviewModelPlayer()
{
}

#if ENABLE(MODEL_PROCESS)
WebCore::ModelPlayerIdentifier ARKitInlinePreviewModelPlayer::identifier() const
{
    return m_id;
}
#endif

void ARKitInlinePreviewModelPlayer::load(WebCore::Model&, WebCore::LayoutSize)
{
}

void ARKitInlinePreviewModelPlayer::sizeDidChange(LayoutSize)
{
}

PlatformLayer* ARKitInlinePreviewModelPlayer::layer()
{
    return nullptr;
}

std::optional<WebCore::LayerHostingContextIdentifier> ARKitInlinePreviewModelPlayer::layerHostingContextIdentifier()
{
    return std::nullopt;
}

void ARKitInlinePreviewModelPlayer::enterFullscreen()
{
}

void ARKitInlinePreviewModelPlayer::getCamera(CompletionHandler<void(std::optional<WebCore::HTMLModelElementCamera>&&)>&& completionHandler)
{
    auto modelIdentifier = this->modelIdentifier();
    if (!modelIdentifier) {
        completionHandler(std::nullopt);
        return;
    }

    RefPtr strongPage = m_page.get();
    if (!strongPage) {
        completionHandler(std::nullopt);
        return;
    }

    CompletionHandler<void(Expected<WebCore::HTMLModelElementCamera, WebCore::ResourceError>)> remoteCompletionHandler = [completionHandler = WTFMove(completionHandler)] (Expected<WebCore::HTMLModelElementCamera, WebCore::ResourceError> result) mutable {
        if (!result) {
            completionHandler(std::nullopt);
            return;
        }

        completionHandler(*result);
    };

    strongPage->sendWithAsyncReply(Messages::WebPageProxy::ModelElementGetCamera(*modelIdentifier), WTFMove(remoteCompletionHandler));
}

void ARKitInlinePreviewModelPlayer::setCamera(WebCore::HTMLModelElementCamera camera, CompletionHandler<void(bool success)>&& completionHandler)
{
    auto modelIdentifier = this->modelIdentifier();
    if (!modelIdentifier) {
        completionHandler(false);
        return;
    }

    RefPtr strongPage = m_page.get();
    if (!strongPage) {
        completionHandler(false);
        return;
    }

    CompletionHandler<void(bool)> remoteCompletionHandler = [completionHandler = WTFMove(completionHandler)] (bool success) mutable {
        completionHandler(success);
    };

    strongPage->sendWithAsyncReply(Messages::WebPageProxy::ModelElementSetCamera(*modelIdentifier, camera), WTFMove(remoteCompletionHandler));
}

void ARKitInlinePreviewModelPlayer::isPlayingAnimation(CompletionHandler<void(std::optional<bool>&&)>&& completionHandler)
{
    auto modelIdentifier = this->modelIdentifier();
    if (!modelIdentifier) {
        completionHandler(std::nullopt);
        return;
    }

    RefPtr strongPage = m_page.get();
    if (!strongPage) {
        completionHandler(std::nullopt);
        return;
    }

    CompletionHandler<void(Expected<bool, WebCore::ResourceError>)> remoteCompletionHandler = [completionHandler = WTFMove(completionHandler)] (Expected<bool, WebCore::ResourceError> result) mutable {
        if (!result) {
            completionHandler(std::nullopt);
            return;
        }

        completionHandler(*result);
    };

    strongPage->sendWithAsyncReply(Messages::WebPageProxy::ModelElementIsPlayingAnimation(*modelIdentifier), WTFMove(remoteCompletionHandler));
}

void ARKitInlinePreviewModelPlayer::setAnimationIsPlaying(bool isPlaying, CompletionHandler<void(bool success)>&& completionHandler)
{
    auto modelIdentifier = this->modelIdentifier();
    if (!modelIdentifier) {
        completionHandler(false);
        return;
    }

    RefPtr strongPage = m_page.get();
    if (!strongPage) {
        completionHandler(false);
        return;
    }

    CompletionHandler<void(bool)> remoteCompletionHandler = [completionHandler = WTFMove(completionHandler)] (bool success) mutable {
        completionHandler(success);
    };

    strongPage->sendWithAsyncReply(Messages::WebPageProxy::ModelElementSetAnimationIsPlaying(*modelIdentifier, isPlaying), WTFMove(remoteCompletionHandler));
}

void ARKitInlinePreviewModelPlayer::isLoopingAnimation(CompletionHandler<void(std::optional<bool>&&)>&& completionHandler)
{
    auto modelIdentifier = this->modelIdentifier();
    if (!modelIdentifier) {
        completionHandler(std::nullopt);
        return;
    }

    RefPtr strongPage = m_page.get();
    if (!strongPage) {
        completionHandler(std::nullopt);
        return;
    }

    CompletionHandler<void(Expected<bool, WebCore::ResourceError>)> remoteCompletionHandler = [completionHandler = WTFMove(completionHandler)] (Expected<bool, WebCore::ResourceError> result) mutable {
        if (!result) {
            completionHandler(std::nullopt);
            return;
        }

        completionHandler(*result);
    };

    strongPage->sendWithAsyncReply(Messages::WebPageProxy::ModelElementIsLoopingAnimation(*modelIdentifier), WTFMove(remoteCompletionHandler));
}

void ARKitInlinePreviewModelPlayer::setIsLoopingAnimation(bool isLooping, CompletionHandler<void(bool success)>&& completionHandler)
{
    auto modelIdentifier = this->modelIdentifier();
    if (!modelIdentifier) {
        completionHandler(false);
        return;
    }

    RefPtr strongPage = m_page.get();
    if (!strongPage) {
        completionHandler(false);
        return;
    }

    CompletionHandler<void(bool)> remoteCompletionHandler = [completionHandler = WTFMove(completionHandler)] (bool success) mutable {
        completionHandler(success);
    };

    strongPage->sendWithAsyncReply(Messages::WebPageProxy::ModelElementSetIsLoopingAnimation(*modelIdentifier, isLooping), WTFMove(remoteCompletionHandler));
}

void ARKitInlinePreviewModelPlayer::animationDuration(CompletionHandler<void(std::optional<Seconds>&&)>&& completionHandler)
{
    auto modelIdentifier = this->modelIdentifier();
    if (!modelIdentifier) {
        completionHandler(std::nullopt);
        return;
    }

    RefPtr strongPage = m_page.get();
    if (!strongPage) {
        completionHandler(std::nullopt);
        return;
    }

    CompletionHandler<void(Expected<Seconds, WebCore::ResourceError>)> remoteCompletionHandler = [completionHandler = WTFMove(completionHandler)] (Expected<Seconds, WebCore::ResourceError> result) mutable {
        if (!result) {
            completionHandler(std::nullopt);
            return;
        }

        completionHandler(*result);
    };

    strongPage->sendWithAsyncReply(Messages::WebPageProxy::ModelElementAnimationDuration(*modelIdentifier), WTFMove(remoteCompletionHandler));
}

void ARKitInlinePreviewModelPlayer::animationCurrentTime(CompletionHandler<void(std::optional<Seconds>&&)>&& completionHandler)
{
    auto modelIdentifier = this->modelIdentifier();
    if (!modelIdentifier) {
        completionHandler(std::nullopt);
        return;
    }

    RefPtr strongPage = m_page.get();
    if (!strongPage) {
        completionHandler(std::nullopt);
        return;
    }

    CompletionHandler<void(Expected<Seconds, WebCore::ResourceError>)> remoteCompletionHandler = [completionHandler = WTFMove(completionHandler)] (Expected<Seconds, WebCore::ResourceError> result) mutable {
        if (!result) {
            completionHandler(std::nullopt);
            return;
        }

        completionHandler(*result);
    };

    strongPage->sendWithAsyncReply(Messages::WebPageProxy::ModelElementAnimationCurrentTime(*modelIdentifier), WTFMove(remoteCompletionHandler));
}

void ARKitInlinePreviewModelPlayer::setAnimationCurrentTime(Seconds currentTime, CompletionHandler<void(bool success)>&& completionHandler)
{
    auto modelIdentifier = this->modelIdentifier();
    if (!modelIdentifier) {
        completionHandler(false);
        return;
    }

    RefPtr strongPage = m_page.get();
    if (!strongPage) {
        completionHandler(false);
        return;
    }

    CompletionHandler<void(bool)> remoteCompletionHandler = [completionHandler = WTFMove(completionHandler)] (bool success) mutable {
        completionHandler(success);
    };

    strongPage->sendWithAsyncReply(Messages::WebPageProxy::ModelElementSetAnimationCurrentTime(*modelIdentifier, currentTime), WTFMove(remoteCompletionHandler));
}

void ARKitInlinePreviewModelPlayer::hasAudio(CompletionHandler<void(std::optional<bool>&&)>&& completionHandler)
{
    auto modelIdentifier = this->modelIdentifier();
    if (!modelIdentifier) {
        completionHandler(std::nullopt);
        return;
    }

    RefPtr strongPage = m_page.get();
    if (!strongPage) {
        completionHandler(std::nullopt);
        return;
    }

    CompletionHandler<void(Expected<bool, WebCore::ResourceError>)> remoteCompletionHandler = [completionHandler = WTFMove(completionHandler)] (Expected<bool, WebCore::ResourceError> result) mutable {
        if (!result) {
            completionHandler(std::nullopt);
            return;
        }

        completionHandler(*result);
    };

    strongPage->sendWithAsyncReply(Messages::WebPageProxy::ModelElementHasAudio(*modelIdentifier), WTFMove(remoteCompletionHandler));
}

void ARKitInlinePreviewModelPlayer::isMuted(CompletionHandler<void(std::optional<bool>&&)>&& completionHandler)
{
    auto modelIdentifier = this->modelIdentifier();
    if (!modelIdentifier) {
        completionHandler(std::nullopt);
        return;
    }

    RefPtr strongPage = m_page.get();
    if (!strongPage) {
        completionHandler(std::nullopt);
        return;
    }

    CompletionHandler<void(Expected<bool, WebCore::ResourceError>)> remoteCompletionHandler = [completionHandler = WTFMove(completionHandler)] (Expected<bool, WebCore::ResourceError> result) mutable {
        if (!result) {
            completionHandler(std::nullopt);
            return;
        }

        completionHandler(*result);
    };

    strongPage->sendWithAsyncReply(Messages::WebPageProxy::ModelElementIsMuted(*modelIdentifier), WTFMove(remoteCompletionHandler));
}

void ARKitInlinePreviewModelPlayer::setIsMuted(bool isMuted, CompletionHandler<void(bool success)>&& completionHandler)
{
    auto modelIdentifier = this->modelIdentifier();
    if (!modelIdentifier) {
        completionHandler(false);
        return;
    }

    RefPtr strongPage = m_page.get();
    if (!strongPage) {
        completionHandler(false);
        return;
    }

    CompletionHandler<void(bool)> remoteCompletionHandler = [completionHandler = WTFMove(completionHandler)] (bool success) mutable {
        completionHandler(success);
    };

    strongPage->sendWithAsyncReply(Messages::WebPageProxy::ModelElementSetIsMuted(*modelIdentifier, isMuted), WTFMove(remoteCompletionHandler));
}

Vector<RetainPtr<id>> ARKitInlinePreviewModelPlayer::accessibilityChildren()
{
    // FIXME: https://webkit.org/b/233575 Need to return something to create a remote element connection to the InlinePreviewModel hosted in another process.
    return { };
}

}

#endif

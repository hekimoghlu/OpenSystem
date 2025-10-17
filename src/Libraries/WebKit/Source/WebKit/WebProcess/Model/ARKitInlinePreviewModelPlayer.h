/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 2, 2023.
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

#if ENABLE(ARKIT_INLINE_PREVIEW)

#import "ModelIdentifier.h"
#import "WebPage.h"
#import "WebPageProxyMessages.h"
#import <WebCore/ModelPlayer.h>
#import <WebCore/ModelPlayerClient.h>
#import <wtf/Compiler.h>

namespace WebKit {

class ARKitInlinePreviewModelPlayer : public WebCore::ModelPlayer, public CanMakeWeakPtr<ARKitInlinePreviewModelPlayer> {
public:
    virtual ~ARKitInlinePreviewModelPlayer();

protected:
    explicit ARKitInlinePreviewModelPlayer(WebPage&, WebCore::ModelPlayerClient&);

    WebPage* page() { return m_page.get(); }
    WebCore::ModelPlayerClient* client() { return m_client.get(); }

    virtual std::optional<ModelIdentifier> modelIdentifier() = 0;
#if ENABLE(MODEL_PROCESS)
    WebCore::ModelPlayerIdentifier identifier() const final;
#endif

private:
    // WebCore::ModelPlayer overrides.
    void load(WebCore::Model&, WebCore::LayoutSize) override;
    void sizeDidChange(WebCore::LayoutSize) override;
    PlatformLayer* layer() override;
    std::optional<WebCore::LayerHostingContextIdentifier> layerHostingContextIdentifier() override;
    void enterFullscreen() override;
    void getCamera(CompletionHandler<void(std::optional<WebCore::HTMLModelElementCamera>&&)>&&) override;
    void setCamera(WebCore::HTMLModelElementCamera, CompletionHandler<void(bool success)>&&) override;
    void isPlayingAnimation(CompletionHandler<void(std::optional<bool>&&)>&&) override;
    void setAnimationIsPlaying(bool, CompletionHandler<void(bool success)>&&) override;
    void isLoopingAnimation(CompletionHandler<void(std::optional<bool>&&)>&&) override;
    void setIsLoopingAnimation(bool, CompletionHandler<void(bool success)>&&) override;
    void animationDuration(CompletionHandler<void(std::optional<Seconds>&&)>&&) override;
    void animationCurrentTime(CompletionHandler<void(std::optional<Seconds>&&)>&&) override;
    void setAnimationCurrentTime(Seconds, CompletionHandler<void(bool success)>&&) override;
    void hasAudio(CompletionHandler<void(std::optional<bool>&&)>&&) override;
    void isMuted(CompletionHandler<void(std::optional<bool>&&)>&&) override;
    void setIsMuted(bool, CompletionHandler<void(bool success)>&&) override;
    Vector<RetainPtr<id>> accessibilityChildren() override;

    WeakPtr<WebPage> m_page;
    WeakPtr<WebCore::ModelPlayerClient> m_client;
#if ENABLE(MODEL_PROCESS)
    WebCore::ModelPlayerIdentifier m_id;
#endif
};

}

#endif

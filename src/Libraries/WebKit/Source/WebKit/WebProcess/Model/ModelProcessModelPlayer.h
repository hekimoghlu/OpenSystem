/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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

#if ENABLE(MODEL_PROCESS)

#import "ModelIdentifier.h"
#import "WebPage.h"
#import "WebPageProxyMessages.h"
#import <WebCore/ModelPlayer.h>
#import <WebCore/ModelPlayerClient.h>
#import <WebCore/ModelPlayerIdentifier.h>
#import <wtf/Compiler.h>

namespace WebKit {

class ModelProcessModelPlayer
    : public WebCore::ModelPlayer
    , public IPC::MessageReceiver {
public:
    static Ref<ModelProcessModelPlayer> create(WebCore::ModelPlayerIdentifier, WebPage&, WebCore::ModelPlayerClient&);
    virtual ~ModelProcessModelPlayer();

    void ref() const final { WebCore::ModelPlayer::ref(); }
    void deref() const final { WebCore::ModelPlayer::deref(); }

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    std::optional<WebCore::LayerHostingContextIdentifier> layerHostingContextIdentifier() { return m_layerHostingContextIdentifier; };

private:
    explicit ModelProcessModelPlayer(WebCore::ModelPlayerIdentifier, WebPage&, WebCore::ModelPlayerClient&);

    WebPage* page() { return m_page.get(); }
    WebCore::ModelPlayerClient* client() { return m_client.get(); }

    template<typename T> void send(T&& message);
    template<typename T, typename C> void sendWithAsyncReply(T&& message, C&& completionHandler);

    bool modelProcessEnabled() const;

    // Messages
    void didCreateLayer(WebCore::LayerHostingContextIdentifier);
    void didFinishLoading(const WebCore::FloatPoint3D&, const WebCore::FloatPoint3D&);
    void didUpdateEntityTransform(const WebCore::TransformationMatrix&);
    void didUpdateAnimationPlaybackState(bool isPaused, double playbackRate, Seconds duration, Seconds currentTime, MonotonicTime clockTimestamp);
    void didFinishEnvironmentMapLoading(bool succeeded);

    // WebCore::ModelPlayer overrides.
    WebCore::ModelPlayerIdentifier identifier() const final { return m_id; }
    void load(WebCore::Model&, WebCore::LayoutSize) final;
    void sizeDidChange(WebCore::LayoutSize) final;
    PlatformLayer* layer() final;
    void handleMouseDown(const WebCore::LayoutPoint&, MonotonicTime) final;
    void handleMouseMove(const WebCore::LayoutPoint&, MonotonicTime) final;
    void handleMouseUp(const WebCore::LayoutPoint&, MonotonicTime) final;
    void setBackgroundColor(WebCore::Color) final;
    void setEntityTransform(WebCore::TransformationMatrix) final;
    bool supportsTransform(WebCore::TransformationMatrix) final;
    void enterFullscreen() final;
    void getCamera(CompletionHandler<void(std::optional<WebCore::HTMLModelElementCamera>&&)>&&) final;
    void setCamera(WebCore::HTMLModelElementCamera, CompletionHandler<void(bool success)>&&) final;
    void isPlayingAnimation(CompletionHandler<void(std::optional<bool>&&)>&&) final;
    void setAnimationIsPlaying(bool, CompletionHandler<void(bool success)>&&) final;
    void isLoopingAnimation(CompletionHandler<void(std::optional<bool>&&)>&&) final;
    void setIsLoopingAnimation(bool, CompletionHandler<void(bool success)>&&) final;
    void animationDuration(CompletionHandler<void(std::optional<Seconds>&&)>&&) final;
    void animationCurrentTime(CompletionHandler<void(std::optional<Seconds>&&)>&&) final;
    void setAnimationCurrentTime(Seconds, CompletionHandler<void(bool success)>&&) final;
    void hasAudio(CompletionHandler<void(std::optional<bool>&&)>&&) final;
    void isMuted(CompletionHandler<void(std::optional<bool>&&)>&&) final;
    void setIsMuted(bool, CompletionHandler<void(bool success)>&&) final;
    Vector<RetainPtr<id>> accessibilityChildren() final;
    void setAutoplay(bool) final;
    void setLoop(bool) final;
    void setPlaybackRate(double, CompletionHandler<void(double effectivePlaybackRate)>&&) final;
    double duration() const final;
    bool paused() const final;
    void setPaused(bool, CompletionHandler<void(bool succeeded)>&&) final;
    Seconds currentTime() const final;
    void setCurrentTime(Seconds, CompletionHandler<void()>&&) final;
    void setEnvironmentMap(Ref<WebCore::SharedBuffer>&& data) final;
    void setHasPortal(bool) final;

    WebCore::ModelPlayerIdentifier m_id;
    WeakPtr<WebPage> m_page;
    WeakPtr<WebCore::ModelPlayerClient> m_client;

    std::optional<WebCore::LayerHostingContextIdentifier> m_layerHostingContextIdentifier;

    bool m_hasPortal { true };
    bool m_autoplay { false };
    bool m_loop { false };
    double m_requestedPlaybackRate { 1.0 };
    std::optional<double> m_effectivePlaybackRate;
    Seconds m_duration { 0_s };
    bool m_paused { true };
    std::optional<Seconds> m_pendingCurrentTime;
    std::optional<MonotonicTime> m_clockTimestampOfLastCurrentTimeSet;
    std::optional<Seconds> m_lastCachedCurrentTime;
    std::optional<MonotonicTime> m_lastCachedClockTimestamp;
};

}

#endif // ENABLE(MODEL_PROCESS)

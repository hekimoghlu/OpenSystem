/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 20, 2024.
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

#include "ModelPlayer.h"
#include "ModelPlayerClient.h"
#include <wtf/Forward.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class WEBCORE_EXPORT DummyModelPlayer final : public ModelPlayer {
public:
    static Ref<DummyModelPlayer> create(ModelPlayerClient&);
    virtual ~DummyModelPlayer();

private:
    DummyModelPlayer(ModelPlayerClient&);

    // ModelPlayer overrides.
#if ENABLE(MODEL_PROCESS)
    ModelPlayerIdentifier identifier() const final { return m_id; }
#endif
    void load(Model&, LayoutSize) override;
    void sizeDidChange(LayoutSize) override;
    PlatformLayer* layer() override;
    std::optional<LayerHostingContextIdentifier> layerHostingContextIdentifier() override;
    void enterFullscreen() override;
    void handleMouseDown(const LayoutPoint&, MonotonicTime) override;
    void handleMouseMove(const LayoutPoint&, MonotonicTime) override;
    void handleMouseUp(const LayoutPoint&, MonotonicTime) override;
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
#if PLATFORM(COCOA)
    Vector<RetainPtr<id>> accessibilityChildren() override;
#endif

    WeakPtr<ModelPlayerClient> m_client;
#if ENABLE(MODEL_PROCESS)
    ModelPlayerIdentifier m_id;
#endif
};

}

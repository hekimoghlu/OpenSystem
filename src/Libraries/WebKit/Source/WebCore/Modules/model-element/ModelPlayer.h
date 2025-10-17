/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 29, 2024.
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

#include "HTMLModelElementCamera.h"
#include "LayerHostingContextIdentifier.h"
#include "LayoutPoint.h"
#include "LayoutSize.h"
#include "PlatformLayer.h"
#include <optional>
#include <wtf/Forward.h>
#include <wtf/MonotonicTime.h>
#include <wtf/Seconds.h>
#include <wtf/TZoneMalloc.h>

#if ENABLE(MODEL_PROCESS)
#include "ModelPlayerIdentifier.h"
#endif

namespace WebCore {

class Color;
class Model;
class SharedBuffer;
class TransformationMatrix;

class WEBCORE_EXPORT ModelPlayer : public RefCounted<ModelPlayer> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ModelPlayer, WEBCORE_EXPORT);
public:
    virtual ~ModelPlayer();

#if ENABLE(MODEL_PROCESS)
    virtual ModelPlayerIdentifier identifier() const = 0;
#endif

    virtual void load(Model&, LayoutSize) = 0;
    virtual void sizeDidChange(LayoutSize) = 0;
    virtual PlatformLayer* layer() = 0;
    virtual std::optional<LayerHostingContextIdentifier> layerHostingContextIdentifier() = 0;
    virtual void setBackgroundColor(Color);
    virtual void setEntityTransform(TransformationMatrix);
    virtual void enterFullscreen() = 0;
    virtual bool supportsMouseInteraction();
    virtual bool supportsDragging();
    virtual bool supportsTransform(TransformationMatrix);
    virtual void setInteractionEnabled(bool);
    virtual void handleMouseDown(const LayoutPoint&, MonotonicTime) = 0;
    virtual void handleMouseMove(const LayoutPoint&, MonotonicTime) = 0;
    virtual void handleMouseUp(const LayoutPoint&, MonotonicTime) = 0;
    virtual void getCamera(CompletionHandler<void(std::optional<HTMLModelElementCamera>&&)>&&) = 0;
    virtual void setCamera(HTMLModelElementCamera, CompletionHandler<void(bool success)>&&) = 0;
    virtual void isPlayingAnimation(CompletionHandler<void(std::optional<bool>&&)>&&) = 0;
    virtual void setAnimationIsPlaying(bool, CompletionHandler<void(bool success)>&&) = 0;
    virtual void isLoopingAnimation(CompletionHandler<void(std::optional<bool>&&)>&&) = 0;
    virtual void setIsLoopingAnimation(bool, CompletionHandler<void(bool success)>&&) = 0;
    virtual void animationDuration(CompletionHandler<void(std::optional<Seconds>&&)>&&) = 0;
    virtual void animationCurrentTime(CompletionHandler<void(std::optional<Seconds>&&)>&&) = 0;
    virtual void setAnimationCurrentTime(Seconds, CompletionHandler<void(bool success)>&&) = 0;
    virtual void hasAudio(CompletionHandler<void(std::optional<bool>&&)>&&) = 0;
    virtual void isMuted(CompletionHandler<void(std::optional<bool>&&)>&&) = 0;
    virtual void setIsMuted(bool, CompletionHandler<void(bool success)>&&) = 0;
    virtual String inlinePreviewUUIDForTesting() const;
#if PLATFORM(COCOA)
    virtual Vector<RetainPtr<id>> accessibilityChildren() = 0;
#endif
#if ENABLE(MODEL_PROCESS)
    virtual void setAutoplay(bool);
    virtual void setLoop(bool);
    virtual void setPlaybackRate(double, CompletionHandler<void(double effectivePlaybackRate)>&&);
    virtual double duration() const;
    virtual bool paused() const;
    virtual void setPaused(bool, CompletionHandler<void(bool succeeded)>&&);
    virtual Seconds currentTime() const;
    virtual void setCurrentTime(Seconds, CompletionHandler<void()>&&);
    virtual void setEnvironmentMap(Ref<SharedBuffer>&& data);
    virtual void setHasPortal(bool);
#endif
};

}

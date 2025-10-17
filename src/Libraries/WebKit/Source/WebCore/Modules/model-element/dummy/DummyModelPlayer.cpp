/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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
#include "DummyModelPlayer.h"

#include "Model.h"
#include "ResourceError.h"

namespace WebCore {

Ref<DummyModelPlayer> DummyModelPlayer::create(ModelPlayerClient& client)
{
    return adoptRef(*new DummyModelPlayer(client));
}

DummyModelPlayer::DummyModelPlayer(ModelPlayerClient& client)
    : m_client { client }
#if ENABLE(MODEL_PROCESS)
    , m_id(ModelPlayerIdentifier::generate())
#endif
{
}

DummyModelPlayer::~DummyModelPlayer() = default;

void DummyModelPlayer::load(Model& model, LayoutSize)
{
    if (m_client)
        m_client->didFailLoading(*this, ResourceError { errorDomainWebKitInternal, 0, model.url(), "Trying to load model via DummyModelPlayer"_s });
}

PlatformLayer* DummyModelPlayer::layer()
{
    return nullptr;
}

std::optional<LayerHostingContextIdentifier> DummyModelPlayer::layerHostingContextIdentifier()
{
    return std::nullopt;
}

void DummyModelPlayer::sizeDidChange(LayoutSize)
{
}

void DummyModelPlayer::enterFullscreen()
{
}

void DummyModelPlayer::handleMouseDown(const LayoutPoint&, MonotonicTime)
{
}

void DummyModelPlayer::handleMouseMove(const LayoutPoint&, MonotonicTime)
{
}

void DummyModelPlayer::handleMouseUp(const LayoutPoint&, MonotonicTime)
{
}

void DummyModelPlayer::getCamera(CompletionHandler<void(std::optional<WebCore::HTMLModelElementCamera>&&)>&&)
{
}

void DummyModelPlayer::setCamera(WebCore::HTMLModelElementCamera, CompletionHandler<void(bool success)>&&)
{
}

void DummyModelPlayer::isPlayingAnimation(CompletionHandler<void(std::optional<bool>&&)>&&)
{
}

void DummyModelPlayer::setAnimationIsPlaying(bool, CompletionHandler<void(bool success)>&&)
{
}

void DummyModelPlayer::isLoopingAnimation(CompletionHandler<void(std::optional<bool>&&)>&&)
{
}

void DummyModelPlayer::setIsLoopingAnimation(bool, CompletionHandler<void(bool success)>&&)
{
}

void DummyModelPlayer::animationDuration(CompletionHandler<void(std::optional<Seconds>&&)>&&)
{
}

void DummyModelPlayer::animationCurrentTime(CompletionHandler<void(std::optional<Seconds>&&)>&&)
{
}

void DummyModelPlayer::setAnimationCurrentTime(Seconds, CompletionHandler<void(bool success)>&&)
{
}

void DummyModelPlayer::hasAudio(CompletionHandler<void(std::optional<bool>&&)>&&)
{
}

void DummyModelPlayer::isMuted(CompletionHandler<void(std::optional<bool>&&)>&&)
{
}

void DummyModelPlayer::setIsMuted(bool, CompletionHandler<void(bool success)>&&)
{
}

#if PLATFORM(COCOA)
Vector<RetainPtr<id>> DummyModelPlayer::accessibilityChildren()
{
    return { };
}
#endif

}

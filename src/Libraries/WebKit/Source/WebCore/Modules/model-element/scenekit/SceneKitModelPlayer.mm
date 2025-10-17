/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 10, 2023.
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

#if HAVE(SCENEKIT)

#import "SceneKitModelPlayer.h"

#import "SceneKitModel.h"
#import "SceneKitModelLoader.h"
#import <pal/spi/cocoa/SceneKitSPI.h>
#import <wtf/cocoa/VectorCocoa.h>

static std::optional<RetainPtr<id>> makeVectorElement(const RetainPtr<id>*, id arrayElement)
{
    return { retainPtr(arrayElement) };
}

namespace WebCore {

Ref<SceneKitModelPlayer> SceneKitModelPlayer::create(ModelPlayerClient& client)
{
    return adoptRef(*new SceneKitModelPlayer(client));
}

SceneKitModelPlayer::SceneKitModelPlayer(ModelPlayerClient& client)
    : m_client { client }
    , m_layer { adoptNS([[SCNMetalLayer alloc] init]) }
{
    m_layer.get().autoenablesDefaultLighting = YES;

    // FIXME: This should be done by the caller.
    m_layer.get().contentsScale = 2.0;
}

SceneKitModelPlayer::~SceneKitModelPlayer()
{
    // If there is an outstanding load, as indicated by a non-null m_loader, cancel it.
    if (m_loader)
        m_loader->cancel();
}

// MARK: - ModelPlayer overrides.

void SceneKitModelPlayer::load(Model& modelSource, LayoutSize)
{
    if (m_loader)
        m_loader->cancel();

    m_loader = loadSceneKitModel(modelSource, *this);
}

void SceneKitModelPlayer::sizeDidChange(LayoutSize)
{
}

PlatformLayer* SceneKitModelPlayer::layer()
{
    return m_layer.get();
}

std::optional<LayerHostingContextIdentifier> SceneKitModelPlayer::layerHostingContextIdentifier()
{
    return std::nullopt;
}

void SceneKitModelPlayer::enterFullscreen()
{
}

void SceneKitModelPlayer::handleMouseDown(const LayoutPoint&, MonotonicTime)
{
}

void SceneKitModelPlayer::handleMouseMove(const LayoutPoint&, MonotonicTime)
{
}

void SceneKitModelPlayer::handleMouseUp(const LayoutPoint&, MonotonicTime)
{
}

void SceneKitModelPlayer::getCamera(CompletionHandler<void(std::optional<HTMLModelElementCamera>&&)>&&)
{
}

void SceneKitModelPlayer::setCamera(HTMLModelElementCamera, CompletionHandler<void(bool success)>&&)
{
}

void SceneKitModelPlayer::isPlayingAnimation(CompletionHandler<void(std::optional<bool>&&)>&&)
{
}

void SceneKitModelPlayer::setAnimationIsPlaying(bool, CompletionHandler<void(bool success)>&&)
{
}

void SceneKitModelPlayer::isLoopingAnimation(CompletionHandler<void(std::optional<bool>&&)>&&)
{
}

void SceneKitModelPlayer::setIsLoopingAnimation(bool, CompletionHandler<void(bool success)>&&)
{
}

void SceneKitModelPlayer::animationDuration(CompletionHandler<void(std::optional<Seconds>&&)>&&)
{
}

void SceneKitModelPlayer::animationCurrentTime(CompletionHandler<void(std::optional<Seconds>&&)>&&)
{
}

void SceneKitModelPlayer::setAnimationCurrentTime(Seconds, CompletionHandler<void(bool success)>&&)
{
}

void SceneKitModelPlayer::hasAudio(CompletionHandler<void(std::optional<bool>&&)>&&)
{
}

void SceneKitModelPlayer::isMuted(CompletionHandler<void(std::optional<bool>&&)>&&)
{
}

void SceneKitModelPlayer::setIsMuted(bool, CompletionHandler<void(bool success)>&&)
{
}

// MARK: - SceneKitModelLoaderClient overrides.

void SceneKitModelPlayer::didFinishLoading(SceneKitModelLoader& loader, Ref<SceneKitModel> model)
{
    dispatch_assert_queue(dispatch_get_main_queue());
    ASSERT_UNUSED(loader, &loader == m_loader.get());

    m_loader = nullptr;
    m_model = WTFMove(model);

    updateScene();

    if (m_client)
        m_client->didFinishLoading(*this);
}

void SceneKitModelPlayer::didFailLoading(SceneKitModelLoader& loader, const ResourceError& error)
{
    dispatch_assert_queue(dispatch_get_main_queue());
    ASSERT_UNUSED(loader, &loader == m_loader.get());

    m_loader = nullptr;

    if (!m_client)
        m_client->didFailLoading(*this, error);
}

void SceneKitModelPlayer::updateScene()
{
    if (m_layer.get().scene == m_model->defaultScene())
        return;
    m_layer.get().scene = m_model->defaultScene();
}

Vector<RetainPtr<id>> SceneKitModelPlayer::accessibilityChildren()
{
#if PLATFORM(IOS_FAMILY)
    NSArray *children = [m_model->defaultScene() accessibilityElements];
#else
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    NSArray *children = [m_model->defaultScene() accessibilityAttributeValue:NSAccessibilityChildrenAttribute];
ALLOW_DEPRECATED_DECLARATIONS_END
#endif
    return makeVector<RetainPtr<id>>(children);
}

#if ENABLE(MODEL_PROCESS)
WebCore::ModelPlayerIdentifier SceneKitModelPlayer::identifier() const
{
    return m_id;
}
#endif

}

#endif

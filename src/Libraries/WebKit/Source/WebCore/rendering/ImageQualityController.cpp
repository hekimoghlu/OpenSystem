/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 20, 2024.
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
#include "ImageQualityController.h"

#include "GraphicsContext.h"
#include "LocalFrame.h"
#include "Page.h"
#include "RenderBoxModelObject.h"
#include "RenderStyleInlines.h"
#include "RenderView.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ImageQualityController);

static const double cInterpolationCutoff = 800. * 800.;
static const Seconds lowQualityTimeThreshold { 500_ms };

ImageQualityController::ImageQualityController(const RenderView& renderView)
    : m_renderView(renderView)
    , m_timer(*this, &ImageQualityController::highQualityRepaintTimerFired, lowQualityTimeThreshold)
{
}

void ImageQualityController::removeLayer(RenderBoxModelObject* object, LayerSizeMap* innerMap, const void* layer)
{
    if (!innerMap)
        return;
    innerMap->remove(layer);
    if (innerMap->isEmpty())
        removeObject(object);
}

void ImageQualityController::set(RenderBoxModelObject* object, LayerSizeMap* innerMap, const void* layer, const LayoutSize& size)
{
    if (innerMap)
        innerMap->set(layer, size);
    else {
        LayerSizeMap newInnerMap;
        newInnerMap.set(layer, size);
        m_objectLayerSizeMap.set(*object, newInnerMap);
    }
}

void ImageQualityController::removeObject(RenderBoxModelObject* object)
{
    m_objectLayerSizeMap.remove(object);
    if (m_objectLayerSizeMap.isEmpty()) {
        m_animatedResizeIsActive = false;
        m_timer.stop();
    }
}

void ImageQualityController::highQualityRepaintTimerFired()
{
    if (m_renderView.renderTreeBeingDestroyed())
        return;
    if (!m_animatedResizeIsActive && !m_liveResizeOptimizationIsActive)
        return;
    m_animatedResizeIsActive = false;

    // If the FrameView is in live resize, punt the timer and hold back for now.
    if (m_renderView.frameView().inLiveResize()) {
        restartTimer();
        return;
    }

    for (auto it = m_objectLayerSizeMap.begin(), end = m_objectLayerSizeMap.end(); it != end; ++it)
        it->key->repaint();

    m_liveResizeOptimizationIsActive = false;
}

void ImageQualityController::restartTimer()
{
    m_timer.restart();
}

std::optional<InterpolationQuality> ImageQualityController::interpolationQualityFromStyle(const RenderStyle& style)
{
    switch (style.imageRendering()) {
    case ImageRendering::OptimizeSpeed:
        return InterpolationQuality::Low;
    case ImageRendering::CrispEdges:
    case ImageRendering::Pixelated:
        return InterpolationQuality::DoNotInterpolate;
    case ImageRendering::OptimizeQuality:
        return InterpolationQuality::Default; // FIXME: CSS 3 Images says that optimizeQuality should behave like 'auto', but that prevents authors from overriding this low quality rendering behavior.
    case ImageRendering::Auto:
        break;
    }
    return std::nullopt;
}

InterpolationQuality ImageQualityController::chooseInterpolationQuality(GraphicsContext& context, RenderBoxModelObject* object, Image& image, const void* layer, const LayoutSize& size)
{
    // If the image is not a bitmap image, then none of this is relevant and we just paint at high quality.
    if (!(image.isBitmapImage() || image.isPDFDocumentImage()) || context.paintingDisabled())
        return InterpolationQuality::Default;

    if (std::optional<InterpolationQuality> styleInterpolation = interpolationQualityFromStyle(object->style()))
        return styleInterpolation.value();

    // Make sure to use the unzoomed image size, since if a full page zoom is in effect, the image
    // is actually being scaled.
    IntSize imageSize(image.width(), image.height());

    // Look ourselves up in the hashtables.
    auto i = m_objectLayerSizeMap.find(object);
    auto* innerMap = i != m_objectLayerSizeMap.end() ? &i->value : 0;
    std::optional<LayoutSize> oldSize;
    if (innerMap) {
        auto j = innerMap->find(layer);
        if (j != innerMap->end())
            oldSize = j->value;
    }

    // If the containing FrameView is being resized, paint at low quality until resizing is finished.
    if (auto* frame = object->document().frame()) {
        bool frameViewIsCurrentlyInLiveResize = frame->view() && frame->view()->inLiveResize();
        if (frameViewIsCurrentlyInLiveResize) {
            set(object, innerMap, layer, size);
            restartTimer();
            m_liveResizeOptimizationIsActive = true;
            return InterpolationQuality::Low;
        }
        if (m_liveResizeOptimizationIsActive)
            return InterpolationQuality::Default;
    }

    auto contextIsScaled = [](GraphicsContext& context) {
        return !context.getCTM().isIdentityOrTranslationOrFlipped();
    };

    if (size == imageSize && !contextIsScaled(context)) {
        // There is no scale in effect. If we had a scale in effect before, we can just remove this object from the list.
        removeLayer(object, innerMap, layer);
        return InterpolationQuality::Default;
    }

    // There is no need to hash scaled images that always use low quality mode when the page demands it. This is the iChat case.
    if (m_renderView.page().inLowQualityImageInterpolationMode()) {
        double totalPixels = static_cast<double>(image.width()) * static_cast<double>(image.height());
        if (totalPixels > cInterpolationCutoff)
            return InterpolationQuality::Low;
    }

    auto saveEntryIfNewOrSizeChanged = [&]() {
        if (!oldSize || oldSize.value() != size)
            set(object, innerMap, layer, size);
    };

    // If an animated resize is active, paint in low quality and kick the timer ahead.
    if (m_animatedResizeIsActive) {
        saveEntryIfNewOrSizeChanged();
        restartTimer();
        return InterpolationQuality::Low;
    }

    // If this is the first time resizing this image, or its size is the
    // same as the last resize, draw at high res, but record the paint
    // size and set the timer.
    if (!oldSize || oldSize.value() == size) {
        saveEntryIfNewOrSizeChanged();
        restartTimer();
        return InterpolationQuality::Default;
    }

    // If the timer is no longer active, draw at high quality and don't set the timer.
    if (!m_timer.isActive()) {
        removeLayer(object, innerMap, layer);
        return InterpolationQuality::Default;
    }

    // This object has been resized to two different sizes while the timer
    // is active, so draw at low quality, set the flag for animated resizes and
    // the object to the list for high quality redraw.
    saveEntryIfNewOrSizeChanged();
    m_animatedResizeIsActive = true;
    restartTimer();
    return InterpolationQuality::Low;
}

}

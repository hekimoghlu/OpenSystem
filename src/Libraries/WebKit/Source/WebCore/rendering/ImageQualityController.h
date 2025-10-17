/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 31, 2022.
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

#include "GraphicsTypes.h"
#include "Timer.h"
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class GraphicsContext;
class Image;
class LayoutSize;
class RenderBoxModelObject;
class RenderView;
class RenderStyle;

class ImageQualityController {
    WTF_MAKE_TZONE_ALLOCATED(ImageQualityController);
    WTF_MAKE_NONCOPYABLE(ImageQualityController);
public:
    explicit ImageQualityController(const RenderView&);

    static std::optional<InterpolationQuality> interpolationQualityFromStyle(const RenderStyle&);
    InterpolationQuality chooseInterpolationQuality(GraphicsContext&, RenderBoxModelObject*, Image&, const void* layer, const LayoutSize&);

    void rendererWillBeDestroyed(RenderBoxModelObject& renderer) { removeObject(&renderer); }

private:
    using LayerSizeMap = UncheckedKeyHashMap<const void*, LayoutSize>;
    using ObjectLayerSizeMap = UncheckedKeyHashMap<SingleThreadWeakRef<RenderBoxModelObject>, LayerSizeMap>;

    void removeLayer(RenderBoxModelObject*, LayerSizeMap* innerMap, const void* layer);
    void set(RenderBoxModelObject*, LayerSizeMap* innerMap, const void* layer, const LayoutSize&);
    void highQualityRepaintTimerFired();
    void restartTimer();
    void removeObject(RenderBoxModelObject*);

    const RenderView& m_renderView;
    ObjectLayerSizeMap m_objectLayerSizeMap;
    DeferrableOneShotTimer m_timer;
    bool m_animatedResizeIsActive { false };
    bool m_liveResizeOptimizationIsActive { false };
};

} // namespace WebCore

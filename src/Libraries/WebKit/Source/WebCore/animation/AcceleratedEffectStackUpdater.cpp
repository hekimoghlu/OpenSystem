/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 23, 2023.
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
#include "AcceleratedEffectStackUpdater.h"

#if ENABLE(THREADED_ANIMATION_RESOLUTION)

#include "AnimationMalloc.h"
#include "Document.h"
#include "LocalDOMWindow.h"
#include "Page.h"
#include "Performance.h"
#include "RenderElement.h"
#include "RenderLayer.h"
#include "RenderLayerBacking.h"
#include "RenderLayerModelObject.h"
#include "RenderStyleConstants.h"
#include "Styleable.h"
#include <wtf/MonotonicTime.h>

namespace WebCore {

AcceleratedEffectStackUpdater::AcceleratedEffectStackUpdater(Document& document)
{
    auto now = MonotonicTime::now();
    m_timeOrigin = now.secondsSinceEpoch();
    if (RefPtr domWindow = document.domWindow())
        m_timeOrigin -= Seconds::fromMilliseconds(domWindow->performance().relativeTimeFromTimeOriginInReducedResolution(now));
}

void AcceleratedEffectStackUpdater::updateEffectStacks()
{
    auto targetsPendingUpdate = std::exchange(m_targetsPendingUpdate, { });
    for (auto [element, pseudoElementIdentifier] : targetsPendingUpdate) {
        if (!element)
            continue;

        Styleable target { *element, pseudoElementIdentifier };

        auto* renderer = dynamicDowncast<RenderLayerModelObject>(target.renderer());
        if (!renderer || !renderer->isComposited())
            continue;

        auto* renderLayer = renderer->layer();
        ASSERT(renderLayer && renderLayer->backing());
        renderLayer->backing()->updateAcceleratedEffectsAndBaseValues();
    }
}

void AcceleratedEffectStackUpdater::updateEffectStackForTarget(const Styleable& target)
{
    m_targetsPendingUpdate.add({ &target.element, target.pseudoElementIdentifier });
}

} // namespace WebCore

#endif // ENABLE(THREADED_ANIMATION_RESOLUTION)

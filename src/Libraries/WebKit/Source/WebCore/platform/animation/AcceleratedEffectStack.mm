/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 28, 2021.
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
#include "AcceleratedEffectStack.h"

#if ENABLE(THREADED_ANIMATION_RESOLUTION)

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(AcceleratedEffectStack);

Ref<AcceleratedEffectStack> AcceleratedEffectStack::create()
{
    return adoptRef(*new AcceleratedEffectStack());
}

AcceleratedEffectStack::AcceleratedEffectStack()
{
}

bool AcceleratedEffectStack::hasEffects() const
{
    return !m_primaryLayerEffects.isEmpty() || !m_backdropLayerEffects.isEmpty();
}

void AcceleratedEffectStack::setEffects(AcceleratedEffects&& effects)
{
    m_primaryLayerEffects.clear();
    m_backdropLayerEffects.clear();

    for (auto& effect : effects) {
        auto& animatedProperties = effect->animatedProperties();

        // If we don't have a keyframe targeting backdrop-filter, we can add the effect
        // as-is to the set of effects targeting the primary layer.
        if (!animatedProperties.contains(AcceleratedEffectProperty::BackdropFilter)) {
            m_primaryLayerEffects.append(effect);
            continue;
        }

        // If the only property targeted is backdrop-filter, we can add the effect
        // as-is to the set of effects targeting the backdrop layer.
        if (animatedProperties.hasExactlyOneBitSet()) {
            m_backdropLayerEffects.append(effect);
            continue;
        }

        // Otherwise, this means we have effects targeting both the primary and backdrop
        // layers, so we must split the effect in two: one for backdrop-filter, and one
        // for all other properties.
        OptionSet<AcceleratedEffectProperty> primaryProperties = animatedProperties - AcceleratedEffectProperty::BackdropFilter;
        m_primaryLayerEffects.append(effect->copyWithProperties(primaryProperties));
        OptionSet<AcceleratedEffectProperty> backdropProperties = { AcceleratedEffectProperty::BackdropFilter };
        m_backdropLayerEffects.append(effect->copyWithProperties(backdropProperties));
    }
}

void AcceleratedEffectStack::setBaseValues(AcceleratedEffectValues&& values)
{
    m_baseValues = WTFMove(values);
}

} // namespace WebCore

#endif // ENABLE(THREADED_ANIMATION_RESOLUTION)

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 11, 2022.
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

#include "CSSPropertyNames.h"
#include "CompositeOperation.h"
#include "IterationCompositeOperation.h"
#include "WebAnimationTypes.h"
#include <wtf/HashSet.h>

namespace WebCore {

class CSSPropertyBlendingClient;
class Document;
class RenderStyle;
class Settings;

class CSSPropertyAnimation {
public:
    static bool isPropertyAnimatable(const AnimatableCSSProperty&);
    static bool isPropertyAdditiveOrCumulative(const AnimatableCSSProperty&);
    static bool propertyRequiresBlendingForAccumulativeIteration(const CSSPropertyBlendingClient&, const AnimatableCSSProperty&, const RenderStyle& a, const RenderStyle& b);
    static bool animationOfPropertyIsAccelerated(const AnimatableCSSProperty&, const Settings&);
    static bool propertiesEqual(const AnimatableCSSProperty&, const RenderStyle& a, const RenderStyle& b, const Document&);
    static bool canPropertyBeInterpolated(const AnimatableCSSProperty&, const RenderStyle& a, const RenderStyle& b, const Document&);
    static CSSPropertyID getPropertyAtIndex(int, std::optional<bool>& isShorthand);
    static std::optional<CSSPropertyID> getAcceleratedPropertyAtIndex(int, const Settings&);
    static int getNumProperties();

    static void blendProperty(const CSSPropertyBlendingClient&, const AnimatableCSSProperty&, RenderStyle& destination, const RenderStyle& from, const RenderStyle& to, double progress, CompositeOperation, IterationCompositeOperation = IterationCompositeOperation::Replace, double currentIteration = 0);
};

} // namespace WebCore

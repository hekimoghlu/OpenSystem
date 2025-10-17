/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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

namespace WebCore {

class CSSToLengthConversionData;
class CSSValue;
class RenderStyle;
class TimingFunction;

namespace CSS {
struct EasingFunction;
}

namespace Style {

CSS::EasingFunction toCSSEasingFunction(const TimingFunction&, const RenderStyle&);

Ref<TimingFunction> createTimingFunction(const CSS::EasingFunction&, const CSSToLengthConversionData&);
Ref<TimingFunction> createTimingFunctionDeprecated(const CSS::EasingFunction&);

RefPtr<TimingFunction> createTimingFunction(const CSSValue&, const CSSToLengthConversionData&);
RefPtr<TimingFunction> createTimingFunctionDeprecated(const CSSValue&);

} // namespace Style
} // namespace WebCore

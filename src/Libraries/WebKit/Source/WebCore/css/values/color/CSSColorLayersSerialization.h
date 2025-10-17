/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 24, 2022.
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

#include "CSSPrimitiveValueMappings.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {
namespace CSS {

// https://drafts.csswg.org/css-color-6/#color-layers
template<typename ColorLayersType>
void serializationForCSSColorLayers(StringBuilder& builder, const ColorLayersType& colorLayers)
{
    builder.append("color-layers("_s);

    if (colorLayers.blendMode != BlendMode::Normal)
        builder.append(nameLiteralForSerialization(toCSSValueID(colorLayers.blendMode)), ", "_s);

    builder.append(interleave(colorLayers.colors, [](auto& builder, auto& color) {
        serializationForCSS(builder, color);
    }, ", "_s));

    builder.append(')');
}

} // namespace CSS
} // namespace WebCore

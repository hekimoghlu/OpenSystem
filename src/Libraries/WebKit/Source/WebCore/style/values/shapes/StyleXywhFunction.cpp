/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 25, 2023.
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
#include "StyleXywhFunction.h"

#include "StylePrimitiveNumericTypes+Conversions.h"
#include "StylePrimitiveNumericTypes+Evaluation.h"

namespace WebCore {
namespace Style {

auto ToStyle<CSS::Xywh>::operator()(const CSS::Xywh& value, const BuilderState& state) -> Inset
{
    auto location = toStyle(value.location, state);
    auto size = toStyle(value.size, state);

    return {
        .insets = {
            location.y(),
            reflectSum(location.x(), size.width()),
            reflectSum(location.y(), size.height()),
            location.x(),
        },
        .radii = toStyle(value.radii, state)
    };
}

auto ToStyle<CSS::XywhFunction>::operator()(const CSS::XywhFunction& value, const BuilderState& state) -> InsetFunction
{
    return { toStyle(value.parameters, state) };
}

} // namespace Style
} // namespace WebCore

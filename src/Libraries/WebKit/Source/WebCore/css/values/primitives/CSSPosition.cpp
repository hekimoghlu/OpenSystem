/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 5, 2024.
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
#include "CSSPosition.h"

namespace WebCore {
namespace CSS {

bool isCenterPosition(const Position& position)
{
    auto isCenter = [](const auto& component) {
        return WTF::switchOn(component.offset,
            [](auto)            { return false; },
            [](Keyword::Center) { return true;  },
            [](const LengthPercentage<>& value) {
                return WTF::switchOn(value,
                    [](const LengthPercentage<>::Raw& raw) { return raw == 50_css_percentage; },
                    [](const LengthPercentage<>::Calc&) { return false; }
                );
            }
        );
    };

    return WTF::switchOn(position,
        [&](const TwoComponentPosition& twoComponent) {
            return isCenter(get<0>(twoComponent)) && isCenter(get<1>(twoComponent));
        },
        [&](const FourComponentPosition&) {
            return false;
        }
    );
}

} // namespace CSS
} // namespace WebCore

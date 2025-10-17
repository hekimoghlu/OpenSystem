/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 9, 2022.
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
#include "FloatRoundedRect.h"

#if USE(SKIA)

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <skia/core/SkRRect.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebCore {

FloatRoundedRect::FloatRoundedRect(const SkRRect& skRect)
    : m_rect(skRect.rect())
{
    SkVector corner = skRect.radii(SkRRect::kUpperLeft_Corner);
    m_radii.setTopLeft({ corner.x(), corner.y() });
    corner = skRect.radii(SkRRect::kUpperRight_Corner);
    m_radii.setTopRight({ corner.x(), corner.y() });
    corner = skRect.radii(SkRRect::kLowerRight_Corner);
    m_radii.setBottomRight({ corner.x(), corner.y() });
    corner = skRect.radii(SkRRect::kLowerLeft_Corner);
    m_radii.setBottomLeft({ corner.x(), corner.y() });
}

FloatRoundedRect::operator SkRRect() const
{
    if (!isRounded())
        return SkRRect::MakeRect(rect());

    WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // GLib/Win port
    SkVector radii[4];
    radii[SkRRect::kUpperLeft_Corner].set(m_radii.topLeft().width(), m_radii.topLeft().height());
    radii[SkRRect::kUpperRight_Corner].set(m_radii.topRight().width(), m_radii.topRight().height());
    radii[SkRRect::kLowerRight_Corner].set(m_radii.bottomRight().width(), m_radii.bottomRight().height());
    radii[SkRRect::kLowerLeft_Corner].set(m_radii.bottomLeft().width(), m_radii.bottomLeft().height());
    WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

    SkRRect skRect;
    skRect.setRectRadii(rect(), radii);
    return skRect;
}

} // namespace WebCore

#endif // USE(SKIA)

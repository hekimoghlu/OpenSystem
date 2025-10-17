/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 11, 2022.
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
#include "OffsetRotation.h"

#include "AnimationUtilities.h"
#include <wtf/MathExtras.h>

namespace WebCore {

bool OffsetRotation::canBlend(const OffsetRotation& to) const
{
    return m_hasAuto == to.hasAuto();
}

OffsetRotation OffsetRotation::blend(const OffsetRotation& to, const BlendingContext& context) const
{
    if (context.isDiscrete) {
        ASSERT(!context.progress || context.progress == 1.0);
        return context.progress ? to : *this;
    }

    ASSERT(canBlend(to));
    return OffsetRotation(m_hasAuto, clampTo<float>(WebCore::blend(m_angle, to.angle(), context)));
}

WTF::TextStream& operator<<(WTF::TextStream& ts, const OffsetRotation& rotation)
{
    ts.dumpProperty("angle", rotation.angle());
    ts.dumpProperty("hasAuto", rotation.hasAuto());

    return ts;
}

} // namespace WebCore

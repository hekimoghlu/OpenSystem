/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 9, 2025.
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
#include "ControlPart.h"

#include "FloatRoundedRect.h"
#include "GraphicsContext.h"

namespace WebCore {

ControlPart::ControlPart(StyleAppearance type)
    : m_type(type)
{
}

ControlFactory& ControlPart::controlFactory() const
{
    return m_overrideControlFactory ? *m_overrideControlFactory : ControlFactory::shared();
}

PlatformControl* ControlPart::platformControl() const
{
    if (!m_platformControl)
        m_platformControl = const_cast<ControlPart&>(*this).createPlatformControl();
    return m_platformControl.get();
}

FloatSize ControlPart::sizeForBounds(const FloatRect& bounds, const ControlStyle& style)
{
    auto platformControl = this->platformControl();
    if (!platformControl)
        return bounds.size();

    platformControl->updateCellStates(bounds, style);
    return platformControl->sizeForBounds(bounds, style);
}

FloatRect ControlPart::rectForBounds(const FloatRect& bounds, const ControlStyle& style)
{
    auto platformControl = this->platformControl();
    if (!platformControl)
        return bounds;

    platformControl->updateCellStates(bounds, style);
    return platformControl->rectForBounds(bounds, style);
}

void ControlPart::draw(GraphicsContext& context, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle& style) const
{
    auto platformControl = this->platformControl();
    if (!platformControl)
        return;

    // It's important to get the clip from the context, because it may be significantly
    // smaller than the layer bounds (e.g. tiled layers)
    platformControl->setFocusRingClipRect(context.clipBounds());

    platformControl->updateCellStates(borderRect.rect(), style);
    platformControl->draw(context, borderRect, deviceScaleFactor, style);

    platformControl->setFocusRingClipRect({ });
}

} // namespace WebCore

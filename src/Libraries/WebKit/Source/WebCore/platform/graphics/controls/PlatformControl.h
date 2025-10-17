/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 27, 2024.
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

#include "ControlStyle.h"
#include "FloatRect.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

class ControlPart;
class GraphicsContext;
class FloatRect;
class FloatRoundedRect;

class PlatformControl {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(PlatformControl);

public:
    PlatformControl(ControlPart& owningPart)
        : m_owningPart(owningPart)
    {
    }

    virtual ~PlatformControl() = default;

    virtual void setFocusRingClipRect(const FloatRect&) { }

    virtual void updateCellStates(const FloatRect&, const ControlStyle&) { }

    virtual FloatSize sizeForBounds(const FloatRect& bounds, const ControlStyle&) const { return bounds.size(); }

    virtual FloatRect rectForBounds(const FloatRect& bounds, const ControlStyle&) const { return bounds; }

    virtual void draw(GraphicsContext&, const FloatRoundedRect&, float, const ControlStyle&) { }

protected:
    ControlPart& m_owningPart;
};

} // namespace WebCore

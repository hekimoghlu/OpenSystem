/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 18, 2022.
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
#include "PointerEventsHitRules.h"

namespace WebCore {

PointerEventsHitRules::PointerEventsHitRules(HitTestingTargetType hitTestingTargetType, const HitTestRequest& request, PointerEvents pointerEvents)
    : requireVisible(false)
    , requireFill(false)
    , requireStroke(false)
    , canHitStroke(false)
    , canHitFill(false)
    , canHitBoundingBox(false)
{
    if (request.svgClipContent())
        pointerEvents = PointerEvents::Fill;

    if (hitTestingTargetType == HitTestingTargetType::SVGPath) {
        switch (pointerEvents)
        {
            case PointerEvents::VisiblePainted:
            case PointerEvents::Auto: // "auto" is like "visiblePainted" when in SVG content
                requireFill = true;
                requireStroke = true;
                FALLTHROUGH;
            case PointerEvents::Visible:
                requireVisible = true;
                canHitFill = true;
                canHitStroke = true;
                break;
            case PointerEvents::VisibleFill:
                requireVisible = true;
                canHitFill = true;
                break;
            case PointerEvents::VisibleStroke:
                requireVisible = true;
                canHitStroke = true;
                break;
            case PointerEvents::Painted:
                requireFill = true;
                requireStroke = true;
                FALLTHROUGH;
            case PointerEvents::All:
                canHitFill = true;
                canHitStroke = true;
                break;
            case PointerEvents::Fill:
                canHitFill = true;
                break;
            case PointerEvents::Stroke:
                canHitStroke = true;
                break;
            case PointerEvents::BoundingBox:
                canHitFill = true;
                canHitBoundingBox = true;
                break;
            case PointerEvents::None:
                // nothing to do here, defaults are all false.
                break;
        }
    } else {
        switch (pointerEvents)
        {
            case PointerEvents::VisiblePainted:
            case PointerEvents::Auto: // "auto" is like "visiblePainted" when in SVG content
                requireVisible = true;
                requireFill = true;
                requireStroke = true;
                canHitFill = true;
                canHitStroke = true;
                break;
            case PointerEvents::VisibleFill:
            case PointerEvents::VisibleStroke:
            case PointerEvents::Visible:
                requireVisible = true;
                canHitFill = true;
                canHitStroke = true;
                break;
            case PointerEvents::Painted:
                requireFill = true;
                requireStroke = true;
                canHitFill = true;
                canHitStroke = true;
                break;
            case PointerEvents::Fill:
            case PointerEvents::Stroke:
            case PointerEvents::All:
                canHitFill = true;
                canHitStroke = true;
                break;
            case PointerEvents::BoundingBox:
                canHitFill = true;
                canHitBoundingBox = true;
                break;
            case PointerEvents::None:
                // nothing to do here, defaults are all false.
                break;
        }
    }
}

}

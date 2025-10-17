/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 30, 2023.
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
#include "RenderIFrame.h"

#include "HTMLIFrameElement.h"
#include "HTMLNames.h"
#include "LocalFrame.h"
#include "LocalFrameView.h"
#include "RenderBoxInlines.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderStyleInlines.h"
#include "RenderView.h"
#include "Settings.h"
#include <wtf/StackStats.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderIFrame);

using namespace HTMLNames;
    
RenderIFrame::RenderIFrame(HTMLIFrameElement& element, RenderStyle&& style)
    : RenderFrameBase(Type::IFrame, element, WTFMove(style))
{
    ASSERT(isRenderIFrame());
}

RenderIFrame::~RenderIFrame() = default;

HTMLIFrameElement& RenderIFrame::iframeElement() const
{
    return downcast<HTMLIFrameElement>(RenderFrameBase::frameOwnerElement());
}

bool RenderIFrame::isNonReplacedAtomicInline() const
{
    // FIXME: iFrames should not override this function.
    return isInline();
}

bool RenderIFrame::requiresLayer() const
{
    return RenderFrameBase::requiresLayer() || style().resize() != Resize::None;
}

bool RenderIFrame::isFullScreenIFrame() const
{
    // Some authors implement fullscreen popups as out-of-flow iframes with size set to full viewport (using vw/vh units).
    // The size used may not perfectly match the viewport size so the following heuristic uses a relaxed constraint.
    return style().hasOutOfFlowPosition() && style().usesViewportUnits();
}

void RenderIFrame::layout()
{
    StackStats::LayoutCheckPoint layoutCheckPoint;
    ASSERT(needsLayout());

    updateLogicalWidth();
    // No kids to layout as a replaced element.
    updateLogicalHeight();

    clearOverflow();
    addVisualEffectOverflow();
    updateLayerTransform();

    clearNeedsLayout();
}

}

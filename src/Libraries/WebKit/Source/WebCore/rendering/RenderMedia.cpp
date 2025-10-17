/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 4, 2023.
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

#if ENABLE(VIDEO)
#include "RenderMedia.h"

#include "RenderBoxInlines.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderFragmentedFlow.h"
#include "RenderView.h"
#include <wtf/StackStats.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderMedia);

RenderMedia::RenderMedia(Type type, HTMLMediaElement& element, RenderStyle&& style)
    : RenderImage(type, element, WTFMove(style), ReplacedFlag::IsMedia)
{
    setHasShadowControls(true);
}

RenderMedia::RenderMedia(Type type, HTMLMediaElement& element, RenderStyle&& style, const IntSize& intrinsicSize)
    : RenderImage(type, element, WTFMove(style), ReplacedFlag::IsMedia)
{
    setIntrinsicSize(intrinsicSize);
    setHasShadowControls(true);
}

RenderMedia::~RenderMedia() = default;

void RenderMedia::paintReplaced(PaintInfo&, const LayoutPoint&)
{
}

void RenderMedia::layout()
{
    LayoutSize oldSize = size();
    RenderImage::layout();
    if (oldSize != size())
        mediaElement().layoutSizeChanged();
}

void RenderMedia::styleDidChange(StyleDifference difference, const RenderStyle* oldStyle)
{
    RenderImage::styleDidChange(difference, oldStyle);
    if (!oldStyle || style().usedVisibility() != oldStyle->usedVisibility())
        mediaElement().visibilityDidChange();
}

} // namespace WebCore

#endif

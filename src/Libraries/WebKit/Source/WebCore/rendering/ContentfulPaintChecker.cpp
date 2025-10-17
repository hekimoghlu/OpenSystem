/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 26, 2021.
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
#include "ContentfulPaintChecker.h"

#include "GraphicsContext.h"
#include "LocalFrameView.h"
#include "NullGraphicsContext.h"
#include "RenderView.h"

namespace WebCore {

bool ContentfulPaintChecker::qualifiesForContentfulPaint(LocalFrameView& frameView)
{
    ASSERT(!frameView.needsLayout());
    ASSERT(frameView.renderView());

    auto oldPaintBehavior = frameView.paintBehavior();
    auto oldEntireContents = frameView.paintsEntireContents();

    frameView.setPaintBehavior(PaintBehavior::FlattenCompositingLayers);
    frameView.setPaintsEntireContents(true);

    NullGraphicsContext checkerContext(NullGraphicsContext::PaintInvalidationReasons::DetectingContentfulPaint);
    frameView.paint(checkerContext, frameView.renderView()->documentRect());

    frameView.setPaintsEntireContents(oldEntireContents);
    frameView.setPaintBehavior(oldPaintBehavior);

    return checkerContext.contentfulPaintDetected();
}

}

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 13, 2023.
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
#include "FrameWin.h"

#include "Document.h"
#include "FloatRect.h"
#include "FrameSelection.h"
#include "LocalFrame.h"
#include "LocalFrameView.h"
#include "NotImplemented.h"
#include "PrintContext.h"
#include "RenderObject.h"

namespace WebCore {

GDIObject<HBITMAP> imageFromRect(const LocalFrame*, IntRect&)
{
    notImplemented();
    return GDIObject<HBITMAP>();
}

void computePageRectsForFrame(LocalFrame* frame, const IntRect& printRect, float headerHeight, float footerHeight, float userScaleFactor, Vector<IntRect>& outPages, int& outPageHeight)
{
    PrintContext printContext(frame);
    float pageHeight = 0;
    printContext.computePageRects(printRect, headerHeight, footerHeight, userScaleFactor, pageHeight);
    outPageHeight = static_cast<int>(pageHeight);
    outPages = printContext.pageRects();
}

GDIObject<HBITMAP> imageFromSelection(LocalFrame* frame, bool forceBlackText)
{
    frame->document()->updateLayout();

    frame->view()->setPaintBehavior(OptionSet<PaintBehavior>(PaintBehavior::SelectionOnly) | (forceBlackText ? OptionSet<PaintBehavior>(PaintBehavior::ForceBlackText) : OptionSet<PaintBehavior>()));
    FloatRect fr = frame->selection().selectionBounds();
    IntRect ir(static_cast<int>(fr.x()), static_cast<int>(fr.y()), static_cast<int>(fr.width()), static_cast<int>(fr.height()));
    GDIObject<HBITMAP> image = imageFromRect(frame, ir);
    frame->view()->setPaintBehavior(PaintBehavior::Normal);
    return image;
}

} // namespace WebCore

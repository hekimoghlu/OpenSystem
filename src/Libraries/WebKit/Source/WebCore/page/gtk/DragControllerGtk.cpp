/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 10, 2023.
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
#include "DragController.h"

#include "DataTransfer.h"
#include "Document.h"
#include "DragData.h"
#include "Editor.h"
#include "Element.h"
#include "FrameDestructionObserverInlines.h"
#include "LocalFrame.h"
#include "Pasteboard.h"
#include "markup.h"

namespace WebCore {

// FIXME: These values are straight out of DragControllerMac, so probably have
// little correlation with Gdk standards...
const int DragController::MaxOriginalImageArea = 1500 * 1500;
const int DragController::DragIconRightInset = 7;
const int DragController::DragIconBottomInset = 3;

const float DragController::DragImageAlpha = 0.75f;

bool DragController::isCopyKeyDown(const DragData& dragData)
{
    return dragData.flags().contains(DragApplicationFlags::IsCopyKeyDown);
}

std::optional<DragOperation> DragController::dragOperation(const DragData& dragData)
{
    // FIXME: This logic is incomplete
    if (dragData.containsURL())
        return DragOperation::Copy;

    return std::nullopt;
}

const IntSize& DragController::maxDragImageSize()
{
    static const IntSize maxDragImageSize(200, 200);
    return maxDragImageSize;
}

void DragController::cleanupAfterSystemDrag()
{
}

void DragController::declareAndWriteDragImage(DataTransfer& dataTransfer, Element& element, const URL& url, const String& label)
{
    auto* frame = element.document().frame();
    ASSERT(frame);
    frame->editor().writeImageToPasteboard(dataTransfer.pasteboard(), element, url, label);
}

}

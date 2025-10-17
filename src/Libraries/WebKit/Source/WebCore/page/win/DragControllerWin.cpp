/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 25, 2024.
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
#include "Element.h"
#include "Pasteboard.h"
#include "markup.h"
#include "windows.h"

namespace WebCore {

const int DragController::MaxOriginalImageArea = 1500 * 1500;
const int DragController::DragIconRightInset = 7;
const int DragController::DragIconBottomInset = 3;

const float DragController::DragImageAlpha = 0.75f;

std::optional<DragOperation> DragController::dragOperation(const DragData& dragData)
{
    // FIXME: To match the macOS behaviour we should return std::nullopt.
    // If we are a modal window, we are the drag source, or the window is an attached sheet.
    // If this can be determined from within WebCore operationForDrag can be pulled into
    // WebCore itself.
    if (dragData.containsURL() && !m_didInitiateDrag)
        return DragOperation::Copy;
    return std::nullopt;
}

bool DragController::isCopyKeyDown(const DragData&)
{
    return ::GetAsyncKeyState(VK_CONTROL);
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
    Pasteboard& pasteboard = dataTransfer.pasteboard();

    // FIXME: Do we really need this check?
    if (!pasteboard.writableDataObject())
        return;

    // Order is important here for Explorer's sake
    pasteboard.writeURLToWritableDataObject(url, label);
    pasteboard.writeImageToDataObject(element, url);
    pasteboard.writeMarkup(serializeFragment(element, SerializedNodes::SubtreeIncludingNode, nullptr, ResolveURLs::Yes));
}

}

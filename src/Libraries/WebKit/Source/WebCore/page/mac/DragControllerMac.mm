/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 27, 2023.
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
#import "config.h"
#import "DragController.h"

#if ENABLE(DRAG_SUPPORT)

#import "DataTransfer.h"
#import "DeprecatedGlobalSettings.h"
#import "Document.h"
#import "DocumentFragment.h"
#import "DragClient.h"
#import "DragData.h"
#import "Editor.h"
#import "EditorClient.h"
#import "Element.h"
#import "File.h"
#import "HTMLAttachmentElement.h"
#import "LocalFrame.h"
#import "LocalFrameView.h"
#import "Page.h"
#import "Pasteboard.h"
#import "PasteboardStrategy.h"
#import "PlatformStrategies.h"
#import "Range.h"
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

#if PLATFORM(IOS_FAMILY)
#import <MobileCoreServices/MobileCoreServices.h>
#endif

namespace WebCore {

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
    if (dragData.flags().contains(DragApplicationFlags::IsModal))
        return std::nullopt;

    bool mayContainURL;
    if (canLoadDataFromDraggingPasteboard())
        mayContainURL = dragData.containsURL();
    else
        mayContainURL = dragData.containsURLTypeIdentifier();

    if (!mayContainURL && !dragData.containsPromise())
        return std::nullopt;

    if (!m_documentUnderMouse || (!(dragData.flags().containsAll({ DragApplicationFlags::HasAttachedSheet, DragApplicationFlags::IsSource }))))
        return DragOperation::Copy;

    return std::nullopt;
}

const IntSize& DragController::maxDragImageSize()
{
    static const IntSize maxDragImageSize(400, 400);
    
    return maxDragImageSize;
}

void DragController::cleanupAfterSystemDrag()
{
#if PLATFORM(MAC)
    // Drag has ended, dragEnded *should* have been called, however it is possible
    // for the UIDelegate to take over the drag, and fail to send the appropriate
    // drag termination event.  As dragEnded just resets drag variables, we just
    // call it anyway to be on the safe side.
    // We don't want to do this for WebKit2, since the client call to start the drag
    // is asynchronous.

    if (m_page->mainFrame().virtualView()->platformWidget())
        dragEnded();
#endif
}

#if PLATFORM(IOS_FAMILY)

DragOperation DragController::platformGenericDragOperation()
{
    // On iOS, UIKit skips the -performDrop invocation altogether if MOVE is forbidden.
    // Thus, if MOVE is not allowed in the drag source operation mask, fall back to only other allowable action, COPY.
    return DragOperation::Copy;
}

void DragController::updateSupportedTypeIdentifiersForDragHandlingMethod(DragHandlingMethod dragHandlingMethod, const DragData& dragData) const
{
    Vector<String> supportedTypes;
    switch (dragHandlingMethod) {
    case DragHandlingMethod::PageLoad:
        supportedTypes.append(UTTypeURL.identifier);
        break;
    case DragHandlingMethod::EditPlainText:
        supportedTypes.append(UTTypeURL.identifier);
        supportedTypes.append(UTTypePlainText.identifier);
        break;
    case DragHandlingMethod::EditRichText:
        if (DeprecatedGlobalSettings::attachmentElementEnabled()) {
            supportedTypes.append(WebArchivePboardType);
            supportedTypes.append(UTTypeContent.identifier);
            supportedTypes.append(UTTypeItem.identifier);
        } else {
            for (NSString *type in Pasteboard::supportedWebContentPasteboardTypes())
                supportedTypes.append(type);
        }
        break;
    case DragHandlingMethod::SetColor:
        supportedTypes.append(UIColorPboardType);
        break;
    default:
        for (NSString *type in Pasteboard::supportedFileUploadPasteboardTypes())
            supportedTypes.append(type);
        break;
    }

    auto context = dragData.createPasteboardContext();
    platformStrategies()->pasteboardStrategy()->updateSupportedTypeIdentifiers(supportedTypes, dragData.pasteboardName(), context.get());
}

#endif // PLATFORM(IOS_FAMILY)

void DragController::declareAndWriteDragImage(DataTransfer& dataTransfer, Element& element, const URL& url, const String& label)
{
    client().declareAndWriteDragImage(dataTransfer.pasteboard().name(), element, url, label, element.document().frame());
}

} // namespace WebCore

#endif // ENABLE(DRAG_SUPPORT)

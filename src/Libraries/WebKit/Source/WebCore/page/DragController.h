/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 13, 2024.
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

#include "DragActions.h"
#include "DragImage.h"
#include "IntPoint.h"
#include "IntRect.h"
#include "SimpleRange.h"
#include <wtf/CompletionHandler.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/URL.h>
#include <wtf/WeakRef.h>

namespace WebCore {

class DataTransfer;
class Document;
class DragClient;
class DragData;
class Element;
class FrameSelection;
class HTMLImageElement;
class HTMLInputElement;
class HitTestResult;
class IntRect;
class LocalFrame;
class Page;
class PlatformMouseEvent;

struct DragItem;
struct DragState;
struct PromisedAttachmentInfo;
struct RemoteUserInputEventData;

class DragController {
    WTF_MAKE_TZONE_ALLOCATED(DragController);
    WTF_MAKE_NONCOPYABLE(DragController);
public:
    DragController(Page&, std::unique_ptr<DragClient>&&);
    ~DragController();

    static DragOperation platformGenericDragOperation();

    WEBCORE_EXPORT std::variant<std::optional<DragOperation>, RemoteUserInputEventData> dragEnteredOrUpdated(LocalFrame&, DragData&&);
    WEBCORE_EXPORT void dragExited(LocalFrame&, DragData&&);
    WEBCORE_EXPORT bool performDragOperation(DragData&&);
    WEBCORE_EXPORT void dragCancelled();

    bool mouseIsOverFileInput() const { return m_fileInputElementUnderMouse; }
    unsigned numberOfItemsToBeAccepted() const { return m_numberOfItemsToBeAccepted; }

    // FIXME: It should be possible to remove a number of these accessors once all
    // drag logic is in WebCore.
    void setDidInitiateDrag(bool initiated) { m_didInitiateDrag = initiated; }
    bool didInitiateDrag() const { return m_didInitiateDrag; }
    OptionSet<DragOperation> sourceDragOperationMask() const { return m_sourceDragOperationMask; }
    const URL& draggingImageURL() const { return m_draggingImageURL; }
    void setDragOffset(const IntPoint& offset) { m_dragOffset = offset; }
    const IntPoint& dragOffset() const { return m_dragOffset; }
    OptionSet<DragSourceAction> dragSourceAction() const { return m_dragSourceAction; }
    DragHandlingMethod dragHandlingMethod() const { return m_dragHandlingMethod; }

    Document* documentUnderMouse() const { return m_documentUnderMouse.get(); }
    RefPtr<Document> protectedDocumentUnderMouse() const { return m_documentUnderMouse; }
    OptionSet<DragDestinationAction> dragDestinationActionMask() const { return m_dragDestinationActionMask; }
    OptionSet<DragSourceAction> delegateDragSourceAction(const IntPoint& rootViewPoint);

    RefPtr<Element> draggableElement(const LocalFrame*, Element* start, const IntPoint&, DragState&) const;
    WEBCORE_EXPORT void dragEnded();

    WEBCORE_EXPORT void placeDragCaret(const IntPoint&);

    const Vector<Ref<HTMLImageElement>>& droppedImagePlaceholders() const { return m_droppedImagePlaceholders; }
    const std::optional<SimpleRange>& droppedImagePlaceholderRange() const { return m_droppedImagePlaceholderRange; }

    WEBCORE_EXPORT void finalizeDroppedImagePlaceholder(HTMLImageElement&, CompletionHandler<void()>&&);
    WEBCORE_EXPORT void insertDroppedImagePlaceholdersAtCaret(const Vector<IntSize>& imageSizes);

    void prepareForDragStart(LocalFrame& sourceFrame, OptionSet<DragSourceAction>, Element& sourceElement, DataTransfer&, const IntPoint& dragOrigin) const;
    bool startDrag(LocalFrame& src, const DragState&, OptionSet<DragOperation>, const PlatformMouseEvent& dragEvent, const IntPoint& dragOrigin, HasNonDefaultPasteboardData);
    static const IntSize& maxDragImageSize();

    static const int MaxOriginalImageArea;
    static const int DragIconRightInset;
    static const int DragIconBottomInset;
    static const float DragImageAlpha;

private:
    void updateSupportedTypeIdentifiersForDragHandlingMethod(DragHandlingMethod, const DragData&) const;
    bool dispatchTextInputEventFor(LocalFrame*, const DragData&);
    bool canProcessDrag(const DragData&);
    bool concludeEditDrag(const DragData&);
    std::optional<DragOperation> operationForLoad(const DragData&);
    DragHandlingMethod tryDocumentDrag(LocalFrame&, const DragData&, OptionSet<DragDestinationAction>, std::optional<DragOperation>&);
    bool tryDHTMLDrag(LocalFrame&, const DragData&, std::optional<DragOperation>&);
    std::optional<DragOperation> dragOperation(const DragData&);
    void clearDragCaret();
    bool dragIsMove(FrameSelection&, const DragData&);
    bool isCopyKeyDown(const DragData&);

    void mouseMovedIntoDocument(RefPtr<Document>&&);
    bool shouldUseCachedImageForDragImage(const Image&) const;
    void disallowFileAccessIfNeeded(DragData&);

    std::optional<HitTestResult> hitTestResultForDragStart(LocalFrame&, Element&, const IntPoint&) const;

    void doImageDrag(Element&, const IntPoint&, const IntRect&, LocalFrame&, IntPoint&, const DragState&, PromisedAttachmentInfo&&);
    void doSystemDrag(DragImage, const IntPoint&, const IntPoint&, LocalFrame&, const DragState&, PromisedAttachmentInfo&&);

    void beginDrag(DragItem, LocalFrame&, const IntPoint& mouseDownPoint, const IntPoint& mouseDraggedPoint, DataTransfer&, DragSourceAction);

    bool canLoadDataFromDraggingPasteboard() const
    {
#if PLATFORM(IOS_FAMILY)
        return m_isPerformingDrop;
#else
        return true;
#endif
    }

    DragClient& client() const { return *m_client; }

    bool tryToUpdateDroppedImagePlaceholders(const DragData&);
    void removeAllDroppedImagePlaceholders();

    void cleanupAfterSystemDrag();
    void declareAndWriteDragImage(DataTransfer&, Element&, const URL&, const String& label);
    Ref<Page> protectedPage() const;

    WeakRef<Page> m_page;
    std::unique_ptr<DragClient> m_client;

    RefPtr<Document> m_documentUnderMouse; // The document the mouse was last dragged over.
    RefPtr<Document> m_dragInitiator; // The Document (if any) that initiated the drag.
    RefPtr<HTMLInputElement> m_fileInputElementUnderMouse;
    unsigned m_numberOfItemsToBeAccepted { 0 };
    DragHandlingMethod m_dragHandlingMethod { DragHandlingMethod::None };

    OptionSet<DragDestinationAction> m_dragDestinationActionMask;
    OptionSet<DragSourceAction> m_dragSourceAction;
    bool m_didInitiateDrag { false };
    OptionSet<DragOperation> m_sourceDragOperationMask; // Set in startDrag when a drag starts from a mouse down within WebKit.
    IntPoint m_dragOffset;
    URL m_draggingImageURL;
    bool m_isPerformingDrop { false };
    Vector<Ref<HTMLImageElement>> m_droppedImagePlaceholders;
    std::optional<SimpleRange> m_droppedImagePlaceholderRange;
};

WEBCORE_EXPORT bool isDraggableLink(const Element&);

} // namespace WebCore

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 17, 2023.
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

#include "CachedResourceHandle.h"
#include "DragActions.h"
#include "DragImage.h"
#include <wtf/CheckedRef.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class CachedImage;
class DataTransferItemList;
class Document;
class DragData;
class DragImageLoader;
class Element;
class FileList;
class File;
class Pasteboard;
class ScriptExecutionContext;
enum class WebContentReadingPolicy : bool;

class DataTransfer : public RefCountedAndCanMakeWeakPtr<DataTransfer> {
public:
    // https://html.spec.whatwg.org/multipage/dnd.html#drag-data-store-mode
    enum class StoreMode { Invalid, ReadWrite, Readonly, Protected };

    static Ref<DataTransfer> create();
    static Ref<DataTransfer> createForCopyAndPaste(const Document&, StoreMode, std::unique_ptr<Pasteboard>&&);
    static Ref<DataTransfer> createForInputEvent(const String& plainText, const String& htmlText);

    WEBCORE_EXPORT ~DataTransfer();

    String dropEffect() const;
    void setDropEffect(const String&);

    String effectAllowed() const;
    void setEffectAllowed(const String&);

    DataTransferItemList& items(Document&);
    Vector<String> types(Document&) const;
    Vector<String> typesForItemList(Document&) const;

    FileList& files(Document*) const;
    FileList& files(Document&) const;

    void clearData(const String& type = String());

    String getData(Document&, const String& type) const;
    String getDataForItem(Document&, const String& type) const;

    void setData(Document&, const String& type, const String& data);
    void setDataFromItemList(Document&, const String& type, const String& data);

    void setDragImage(Ref<Element>&&, int x, int y);

    void makeInvalidForSecurity() { m_storeMode = StoreMode::Invalid; }

    bool canReadTypes() const;
    bool canReadData() const;
    bool canWriteData() const;

    bool hasFileOfType(const String&);
    bool hasStringOfType(Document&, const String&);

    Pasteboard& pasteboard() { return *m_pasteboard; }
    void commitToPasteboard(Pasteboard&);

#if ENABLE(DRAG_SUPPORT)
    static Ref<DataTransfer> createForDrag(const Document&);
    static Ref<DataTransfer> createForDragStartEvent(const Document&);
    static Ref<DataTransfer> createForDrop(const Document&, std::unique_ptr<Pasteboard>&&, OptionSet<DragOperation>, bool draggingFiles);
    static Ref<DataTransfer> createForUpdatingDropTarget(const Document&, std::unique_ptr<Pasteboard>&&, OptionSet<DragOperation>, bool draggingFiles);

    bool dropEffectIsUninitialized() const { return m_dropEffect == "uninitialized"_s; }

    OptionSet<DragOperation> sourceOperationMask() const;
    OptionSet<DragOperation> destinationOperationMask() const;
    void setSourceOperationMask(OptionSet<DragOperation>);
    void setDestinationOperationMask(OptionSet<DragOperation>);

    void setDragHasStarted() { m_shouldUpdateDragImage = true; }
    DragImageRef createDragImage(IntPoint& dragLocation) const;
    void updateDragImage();
    RefPtr<Element> dragImageElement() const;

    void moveDragState(Ref<DataTransfer>&&);
    bool hasDragImage() const;

    IntPoint dragLocation() const { return m_dragLocation; }
#endif

    void didAddFileToItemList();
    void updateFileList(ScriptExecutionContext*);

private:
    enum class Type { CopyAndPaste, DragAndDropData, DragAndDropFiles, InputEvent };
    DataTransfer(StoreMode, std::unique_ptr<Pasteboard>, Type = Type::CopyAndPaste, String&& effectAllowed = "uninitialized"_s);

    bool allowsFileAccess() const
    {
        return !forDrag() || forFileDrag();
    }

#if ENABLE(DRAG_SUPPORT)
    bool forDrag() const { return m_type == Type::DragAndDropData || m_type == Type::DragAndDropFiles; }
    bool forFileDrag() const { return m_type == Type::DragAndDropFiles; }
#else
    bool forDrag() const { return false; }
    bool forFileDrag() const { return false; }
#endif

    String readStringFromPasteboard(Document&, const String& lowercaseType, WebContentReadingPolicy) const;
    bool shouldSuppressGetAndSetDataToAvoidExposingFilePaths() const;

    enum class AddFilesType : bool { No, Yes };
    Vector<String> types(Document&, AddFilesType) const;
    Vector<Ref<File>> filesFromPasteboardAndItemList(ScriptExecutionContext*) const;

    String m_originIdentifier;
    StoreMode m_storeMode;
    std::unique_ptr<Pasteboard> m_pasteboard;
    std::unique_ptr<DataTransferItemList> m_itemList;

    mutable RefPtr<FileList> m_fileList;

#if ENABLE(DRAG_SUPPORT)
    Type m_type;
    String m_dropEffect;
    String m_effectAllowed;
    bool m_shouldUpdateDragImage;
    IntPoint m_dragLocation;
    CachedResourceHandle<CachedImage> m_dragImage;
    RefPtr<Element> m_dragImageElement;
    std::unique_ptr<DragImageLoader> m_dragImageLoader;
#endif
};

} // namespace WebCore

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 6, 2024.
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

#include "Color.h"
#include "DragActions.h"
#include "IntPoint.h"
#include "PageIdentifier.h"
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/OptionSet.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

#if PLATFORM(MAC)

#ifdef __OBJC__ 
#import <Foundation/Foundation.h>
#import <AppKit/NSDragging.h>
typedef id <NSDraggingInfo> DragDataRef;
#else
typedef void* DragDataRef;
#endif

#elif PLATFORM(WIN)
typedef struct IDataObject* DragDataRef;
#elif PLATFORM(GTK)
namespace WebCore {
class SelectionData;
}
typedef WebCore::SelectionData* DragDataRef;
#else
typedef void* DragDataRef;
#endif

namespace WebCore {

enum class DragApplicationFlags : uint8_t {
    IsModal = 1,
    IsSource = 2,
    HasAttachedSheet = 4,
    IsCopyKeyDown = 8
};

class PasteboardContext;

#if PLATFORM(WIN)
typedef UncheckedKeyHashMap<unsigned, Vector<String>> DragDataMap;
#endif

class DragData {
public:
    enum FilenameConversionPolicy { DoNotConvertFilenames, ConvertFilenames };
    enum class DraggingPurpose { ForEditing, ForFileUpload, ForColorControl };

    // clientPosition is taken to be the position of the drag event within the target window, with (0,0) at the top left
    WEBCORE_EXPORT DragData(DragDataRef, const IntPoint& clientPosition, const IntPoint& globalPosition, OptionSet<DragOperation>, OptionSet<DragApplicationFlags> = { }, OptionSet<DragDestinationAction> = anyDragDestinationAction(), std::optional<PageIdentifier> pageID = std::nullopt);

    WEBCORE_EXPORT DragData(
#if PLATFORM(COCOA)
        const String& dragStorageName,
#endif
        const IntPoint& clientPosition, const IntPoint& globalPosition, OptionSet<DragOperation>, OptionSet<DragApplicationFlags> = { }, OptionSet<DragDestinationAction> = anyDragDestinationAction(), std::optional<PageIdentifier> pageID = std::nullopt);

#if PLATFORM(COCOA)
    WEBCORE_EXPORT DragData(const String& dragStorageName, const IntPoint& clientPosition, const IntPoint& globalPosition, const Vector<String>&, OptionSet<DragOperation>, OptionSet<DragApplicationFlags> = { }, OptionSet<DragDestinationAction> = anyDragDestinationAction(), std::optional<PageIdentifier> pageID = std::nullopt);
#endif
    // This constructor should used only by WebKit2 IPC because DragData
    // is initialized by the decoder and not in the constructor.
    DragData() = default;
#if PLATFORM(WIN)
    WEBCORE_EXPORT DragData(const DragDataMap&, const IntPoint& clientPosition, const IntPoint& globalPosition, OptionSet<DragOperation> sourceOperationMask, OptionSet<DragApplicationFlags> = { }, std::optional<PageIdentifier> pageID = std::nullopt);
    const DragDataMap& dragDataMap();
    void getDragFileDescriptorData(int& size, String& pathname);
    void getDragFileContentData(int size, void* dataBlob);
#endif
    const IntPoint& clientPosition() const { return m_clientPosition; }
    const IntPoint& globalPosition() const { return m_globalPosition; }
    void setClientPosition(const IntPoint& clientPosition) { m_clientPosition = clientPosition; }
    OptionSet<DragApplicationFlags> flags() const { return m_applicationFlags; }
    DragDataRef platformData() const { return m_platformDragData; }
    OptionSet<DragOperation> draggingSourceOperationMask() const { return m_draggingSourceOperationMask; }
    bool containsURL(FilenameConversionPolicy = ConvertFilenames) const;
    bool containsPlainText() const;
    bool containsCompatibleContent(DraggingPurpose = DraggingPurpose::ForEditing) const;
    String asURL(FilenameConversionPolicy = ConvertFilenames, String* title = nullptr) const;
    String asPlainText() const;
    Vector<String> asFilenames() const;
    Color asColor() const;
    bool canSmartReplace() const;
    bool containsColor() const;
    bool containsFiles() const;
    unsigned numberOfFiles() const;
    OptionSet<DragDestinationAction> dragDestinationActionMask() const { return m_dragDestinationActionMask; }
    void setFileNames(Vector<String>& fileNames) { m_fileNames = WTFMove(fileNames); }
    const Vector<String>& fileNames() const { return m_fileNames; }
    void disallowFileAccess();
#if PLATFORM(COCOA)
    const String& pasteboardName() const { return m_pasteboardName; }
    bool containsURLTypeIdentifier() const;
    bool containsPromise() const;
#endif

    bool shouldMatchStyleOnDrop() const;

    std::optional<PageIdentifier> pageID() const { return m_pageID; }

    std::unique_ptr<PasteboardContext> createPasteboardContext() const;

private:
    IntPoint m_clientPosition;
    IntPoint m_globalPosition;
    DragDataRef m_platformDragData { NULL };
    OptionSet<DragOperation> m_draggingSourceOperationMask;
    OptionSet<DragApplicationFlags> m_applicationFlags;
    Vector<String> m_fileNames;
    OptionSet<DragDestinationAction> m_dragDestinationActionMask;
    std::optional<PageIdentifier> m_pageID;
#if PLATFORM(COCOA)
    String m_pasteboardName;
#endif
#if PLATFORM(WIN)
    DragDataMap m_dragDataMap;
#endif
    bool m_disallowFileAccess { false };
};
    
} // namespace WebCore

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 17, 2025.
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
#include "DragData.h"

#include "COMPtr.h"
#include "ClipboardUtilitiesWin.h"
#include <objidl.h>
#include <pal/text/TextEncoding.h>
#include <shlwapi.h>
#include <wininet.h>
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/RefPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

DragData::DragData(const DragDataMap& data, const IntPoint& clientPosition, const IntPoint& globalPosition, OptionSet<DragOperation> sourceOperationMask, OptionSet<DragApplicationFlags> flags, std::optional<PageIdentifier> pageID)
    : m_clientPosition(clientPosition)
    , m_globalPosition(globalPosition)
    , m_platformDragData(0)
    , m_draggingSourceOperationMask(sourceOperationMask)
    , m_applicationFlags(flags)
    , m_pageID(pageID)
    , m_dragDataMap(data)
{
}

bool DragData::containsURL(FilenameConversionPolicy filenamePolicy) const
{
    if (m_platformDragData)
        return SUCCEEDED(m_platformDragData->QueryGetData(urlWFormat())) 
            || SUCCEEDED(m_platformDragData->QueryGetData(urlFormat()))
            || (filenamePolicy == ConvertFilenames
                && (SUCCEEDED(m_platformDragData->QueryGetData(filenameWFormat()))
                    || SUCCEEDED(m_platformDragData->QueryGetData(filenameFormat()))));
    return m_dragDataMap.contains(urlWFormat()->cfFormat) || m_dragDataMap.contains(urlFormat()->cfFormat)
        || (filenamePolicy == ConvertFilenames && (m_dragDataMap.contains(filenameWFormat()->cfFormat) || m_dragDataMap.contains(filenameFormat()->cfFormat)));
}

const DragDataMap& DragData::dragDataMap()
{
    if (!m_dragDataMap.isEmpty() || !m_platformDragData)
        return m_dragDataMap;
    // Enumerate clipboard content and load it in the map.
    COMPtr<IEnumFORMATETC> itr;

    if (FAILED(m_platformDragData->EnumFormatEtc(DATADIR_GET, &itr)) || !itr)
        return m_dragDataMap;

    FORMATETC dataFormat;
    while (itr->Next(1, &dataFormat, 0) == S_OK) {
        Vector<String> dataStrings;
        getClipboardData(m_platformDragData, &dataFormat, dataStrings);
        if (!dataStrings.isEmpty())
            m_dragDataMap.set(dataFormat.cfFormat, dataStrings); 
    }
    return m_dragDataMap;
}

void DragData::getDragFileDescriptorData(int& size, String& pathname)
{
    size = 0;
    if (m_platformDragData)
        getFileDescriptorData(m_platformDragData, size, pathname);
}

void DragData::getDragFileContentData(int size, void* dataBlob)
{
    if (m_platformDragData)
        getFileContentData(m_platformDragData, size, dataBlob);
}

String DragData::asURL(FilenameConversionPolicy filenamePolicy, String* title) const
{
    return (m_platformDragData) ? getURL(m_platformDragData, filenamePolicy, title) : getURL(&m_dragDataMap, filenamePolicy, title);
}

bool DragData::containsFiles() const
{
    return (m_platformDragData) ? SUCCEEDED(m_platformDragData->QueryGetData(cfHDropFormat())) : m_dragDataMap.contains(cfHDropFormat()->cfFormat);
}

unsigned DragData::numberOfFiles() const
{
    if (!m_platformDragData)
        return 0;

    STGMEDIUM medium;
    if (FAILED(m_platformDragData->GetData(cfHDropFormat(), &medium)))
        return 0;

    HDROP hdrop = static_cast<HDROP>(GlobalLock(medium.hGlobal));

    if (!hdrop)
        return 0;

    unsigned numFiles = DragQueryFileW(hdrop, 0xFFFFFFFF, 0, 0);

    DragFinish(hdrop);
    GlobalUnlock(medium.hGlobal);

    return numFiles;
}

Vector<String> DragData::asFilenames() const
{
    Vector<String> result;

    if (m_platformDragData) {
        WCHAR filename[MAX_PATH];

        STGMEDIUM medium;
        if (FAILED(m_platformDragData->GetData(cfHDropFormat(), &medium)))
            return result;

        HDROP hdrop = reinterpret_cast<HDROP>(GlobalLock(medium.hGlobal)); 

        if (!hdrop)
            return result;

        const unsigned numFiles = DragQueryFileW(hdrop, 0xFFFFFFFF, 0, 0);
        for (unsigned i = 0; i < numFiles; i++) {
            if (!DragQueryFileW(hdrop, i, filename, std::size(filename)))
                continue;
            result.append(filename);
        }

        // Free up memory from drag
        DragFinish(hdrop);

        GlobalUnlock(medium.hGlobal);
        return result;
    }
    result = m_dragDataMap.get(cfHDropFormat()->cfFormat);

    return result;
}

bool DragData::containsPlainText() const
{
    if (m_platformDragData)
        return SUCCEEDED(m_platformDragData->QueryGetData(plainTextWFormat()))
            || SUCCEEDED(m_platformDragData->QueryGetData(plainTextFormat()));
    return m_dragDataMap.contains(plainTextWFormat()->cfFormat) || m_dragDataMap.contains(plainTextFormat()->cfFormat);
}

String DragData::asPlainText() const
{
    return (m_platformDragData) ? getPlainText(m_platformDragData) : getPlainText(&m_dragDataMap);
}

bool DragData::containsColor() const
{
    return false;
}

bool DragData::canSmartReplace() const
{
    if (m_platformDragData)
        return SUCCEEDED(m_platformDragData->QueryGetData(smartPasteFormat()));
    return m_dragDataMap.contains(smartPasteFormat()->cfFormat);
}

bool DragData::containsCompatibleContent(DraggingPurpose) const
{
    return containsPlainText() || containsURL()
        || ((m_platformDragData) ? (containsHTML(m_platformDragData) || containsFilenames(m_platformDragData))
            : (containsHTML(&m_dragDataMap) || containsFilenames(&m_dragDataMap)))
        || containsColor();
}

Color DragData::asColor() const
{
    return Color();
}

bool DragData::shouldMatchStyleOnDrop() const
{
    return false;
}

}

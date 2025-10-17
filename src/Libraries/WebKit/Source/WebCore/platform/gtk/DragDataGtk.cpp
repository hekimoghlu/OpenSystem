/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 2, 2022.
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

#include "SelectionData.h"

namespace WebCore {

bool DragData::canSmartReplace() const
{
    return false;
}

bool DragData::containsColor() const
{
    return false;
}

bool DragData::containsFiles() const
{
    return !m_disallowFileAccess && m_platformDragData->hasFilenames();
}

unsigned DragData::numberOfFiles() const
{
    return !m_disallowFileAccess ? m_platformDragData->filenames().size() : 0;
}

Vector<String> DragData::asFilenames() const
{
    if (m_disallowFileAccess)
        return { };
    return m_platformDragData->filenames();
}

bool DragData::containsPlainText() const
{
    return m_platformDragData->hasText();
}

String DragData::asPlainText() const
{
    return m_platformDragData->text();
}

Color DragData::asColor() const
{
    return Color();
}

bool DragData::containsCompatibleContent(DraggingPurpose) const
{
    return containsPlainText() || containsURL() || m_platformDragData->hasMarkup() || containsColor() || containsFiles();
}

bool DragData::containsURL(FilenameConversionPolicy filenamePolicy) const
{
    return !asURL(filenamePolicy).isEmpty();
}

String DragData::asURL(FilenameConversionPolicy filenamePolicy, String* title) const
{
    if (!m_platformDragData->hasURL())
        return String();
    if (filenamePolicy != ConvertFilenames) {
        if (m_platformDragData->url().protocolIsFile())
            return { };
    }

    if (title)
        *title = m_platformDragData->urlLabel();
    return m_platformDragData->url().string();
}

bool DragData::shouldMatchStyleOnDrop() const
{
    return false;
}

}

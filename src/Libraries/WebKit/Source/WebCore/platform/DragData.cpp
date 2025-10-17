/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 13, 2025.
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
#include "PagePasteboardContext.h"
#include "PlatformEvent.h"
#include "PlatformKeyboardEvent.h"

#if ENABLE(DRAG_SUPPORT)
namespace WebCore {

#if !PLATFORM(COCOA)
DragData::DragData(DragDataRef data, const IntPoint& clientPosition, const IntPoint& globalPosition, OptionSet<DragOperation> sourceOperationMask, OptionSet<DragApplicationFlags> flags, OptionSet<DragDestinationAction> destinationActionMask, std::optional<PageIdentifier> pageID)
    : m_clientPosition(clientPosition)
    , m_globalPosition(globalPosition)
    , m_platformDragData(data)
    , m_draggingSourceOperationMask(sourceOperationMask)
    , m_applicationFlags(flags)
    , m_dragDestinationActionMask(destinationActionMask)
    , m_pageID(pageID)
{  
}

DragData::DragData(const IntPoint& clientPosition, const IntPoint& globalPosition, OptionSet<DragOperation> sourceOperationMask, OptionSet<DragApplicationFlags> flags, OptionSet<DragDestinationAction> destinationActionMask, std::optional<PageIdentifier> pageID)
    : m_clientPosition(clientPosition)
    , m_globalPosition(globalPosition)
    , m_platformDragData(0)
    , m_draggingSourceOperationMask(sourceOperationMask)
    , m_applicationFlags(flags)
    , m_dragDestinationActionMask(destinationActionMask)
    , m_pageID(pageID)
{
}
#endif

std::unique_ptr<PasteboardContext> DragData::createPasteboardContext() const
{
    return PagePasteboardContext::create(std::optional<PageIdentifier> { m_pageID });
}

void DragData::disallowFileAccess()
{
    m_fileNames = { };
    m_disallowFileAccess = true;
}

} // namespace WebCore

#endif // ENABLE(DRAG_SUPPORT)

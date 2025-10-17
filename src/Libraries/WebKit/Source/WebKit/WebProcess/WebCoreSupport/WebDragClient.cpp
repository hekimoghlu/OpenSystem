/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 29, 2023.
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
#include "WebDragClient.h"

#if ENABLE(DRAG_SUPPORT)

#include "WebPage.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebDragClient);

void WebDragClient::willPerformDragDestinationAction(DragDestinationAction action, const DragData&)
{
    if (action == DragDestinationAction::Load)
        m_page->willPerformLoadDragDestinationAction();
    else
        m_page->mayPerformUploadDragDestinationAction(); // Upload can happen from a drop event handler, so we should prepare early.
}

void WebDragClient::willPerformDragSourceAction(DragSourceAction, const IntPoint&, DataTransfer&)
{
}

OptionSet<DragSourceAction> WebDragClient::dragSourceActionMaskForPoint(const IntPoint&)
{
    return m_page->allowedDragSourceActions();
}

#if !PLATFORM(COCOA) && !PLATFORM(GTK)
void WebDragClient::startDrag(DragItem, DataTransfer&, Frame&)
{
}

void WebDragClient::didConcludeEditDrag()
{
}
#endif

} // namespace WebKit

#endif // ENABLE(DRAG_SUPPORT)

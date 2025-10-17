/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 26, 2022.
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
#ifndef WebDragClient_h
#define WebDragClient_h

#if ENABLE(DRAG_SUPPORT)

#include <WebCore/DragClient.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class WebPage;

class WebDragClient : public WebCore::DragClient {
    WTF_MAKE_TZONE_ALLOCATED(WebDragClient);
public:
    WebDragClient(WebPage* page)
        : m_page(page)
    {
    }

private:
    void willPerformDragDestinationAction(WebCore::DragDestinationAction, const WebCore::DragData&) override;
    void willPerformDragSourceAction(WebCore::DragSourceAction, const WebCore::IntPoint&, WebCore::DataTransfer&) override;
    OptionSet<WebCore::DragSourceAction> dragSourceActionMaskForPoint(const WebCore::IntPoint& windowPoint) override;

    void startDrag(WebCore::DragItem, WebCore::DataTransfer&, WebCore::Frame&) override;
    void didConcludeEditDrag() override;

#if PLATFORM(COCOA)
    void declareAndWriteDragImage(const String& pasteboardName, WebCore::Element&, const URL&, const String&, WebCore::LocalFrame*) override;
#endif

    WeakPtr<WebPage> m_page;
};

} // namespace WebKit

#endif // ENABLE(DRAG_SUPPORT)

#endif // WebDragClient_h

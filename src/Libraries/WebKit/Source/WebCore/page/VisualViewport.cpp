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
#include "VisualViewport.h"

#include "Chrome.h"
#include "ChromeClient.h"
#include "ContextDestructionObserver.h"
#include "Document.h"
#include "Event.h"
#include "EventNames.h"
#include "LocalDOMWindow.h"
#include "LocalFrame.h"
#include "LocalFrameView.h"
#include "Page.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(VisualViewport);

VisualViewport::VisualViewport(LocalDOMWindow& window)
    : LocalDOMWindowProperty(&window)
{
}

enum EventTargetInterfaceType VisualViewport::eventTargetInterface() const
{
    return EventTargetInterfaceType::VisualViewport;
}

ScriptExecutionContext* VisualViewport::scriptExecutionContext() const
{
    RefPtr window = this->window();
    return window ? window->document() : nullptr;
}

bool VisualViewport::addEventListener(const AtomString& eventType, Ref<EventListener>&& listener, const AddEventListenerOptions& options)
{
    if (!EventTarget::addEventListener(eventType, WTFMove(listener), options))
        return false;

    if (auto* frame = this->frame())
        frame->document()->addListenerTypeIfNeeded(eventType);
    return true;
}

void VisualViewport::updateFrameLayout() const
{
    ASSERT(frame());
    frame()->document()->updateLayout({ LayoutOptions::IgnorePendingStylesheets, LayoutOptions::RunPostLayoutTasksSynchronously });
}

double VisualViewport::offsetLeft() const
{
    if (!frame())
        return 0;

    updateFrameLayout();
    return m_offsetLeft;
}

double VisualViewport::offsetTop() const
{
    if (!frame())
        return 0;

    updateFrameLayout();
    return m_offsetTop;
}

double VisualViewport::pageLeft() const
{
    if (!frame())
        return 0;

    updateFrameLayout();
    return m_pageLeft;
}

double VisualViewport::pageTop() const
{
    if (!frame())
        return 0;

    updateFrameLayout();
    return m_pageTop;
}

double VisualViewport::width() const
{
    if (!frame())
        return 0;

    updateFrameLayout();
    return m_width;
}

double VisualViewport::height() const
{
    if (!frame())
        return 0;

    updateFrameLayout();
    return m_height;
}

double VisualViewport::scale() const
{
    // Subframes always have scale 1 since they aren't scaled relative to their parent frame.
    auto* frame = this->frame();
    if (!frame || !frame->isMainFrame())
        return 1;

    updateFrameLayout();
    return m_scale;
}

void VisualViewport::update()
{
    double offsetLeft = 0;
    double offsetTop = 0;
    m_pageLeft = 0;
    m_pageTop = 0;
    double width = 0;
    double height = 0;
    double scale = 1;

    RefPtr frame = this->frame();
    if (frame) {
        if (auto* view = frame->view()) {
            auto visualViewportRect = view->visualViewportRect();
            auto layoutViewportRect = view->layoutViewportRect();
            auto pageZoomFactor = frame->pageZoomFactor();
            ASSERT(pageZoomFactor);
            offsetLeft = (visualViewportRect.x() - layoutViewportRect.x()) / pageZoomFactor;
            offsetTop = (visualViewportRect.y() - layoutViewportRect.y()) / pageZoomFactor;
            m_pageLeft = visualViewportRect.x() / pageZoomFactor;
            m_pageTop = visualViewportRect.y() / pageZoomFactor;
            width = visualViewportRect.width() / pageZoomFactor;
            height = visualViewportRect.height() / pageZoomFactor;
        }
        if (RefPtr page = frame->page())
            scale = page->pageScaleFactor() / page->chrome().client().baseViewportLayoutSizeScaleFactor();
    }

    RefPtr<Document> document = frame ? frame->document() : nullptr;
    if (m_offsetLeft != offsetLeft || m_offsetTop != offsetTop) {
        if (document)
            document->setNeedsVisualViewportScrollEvent();
        m_offsetLeft = offsetLeft;
        m_offsetTop = offsetTop;
    }
    if (m_width != width || m_height != height) {
        if (document)
            document->setNeedsVisualViewportResize();
        m_width = width;
        m_height = height;
        m_scale = scale;
    }
}

} // namespace WebCore

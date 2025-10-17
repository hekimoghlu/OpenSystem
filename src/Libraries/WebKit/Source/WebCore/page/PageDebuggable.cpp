/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 12, 2024.
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
#include "PageDebuggable.h"

#if ENABLE(REMOTE_INSPECTOR)

#include "Document.h"
#include "InspectorController.h"
#include "LocalFrame.h"
#include "Page.h"
#include "Settings.h"
#include <JavaScriptCore/InspectorAgentBase.h>
#include <wtf/MainThread.h>
#include <wtf/TZoneMallocInlines.h>


namespace WebCore {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(PageDebuggable);
Ref<PageDebuggable> PageDebuggable::create(Page& page)
{
    return adoptRef(*new PageDebuggable(page));
}

PageDebuggable::PageDebuggable(Page& page)
    : m_page(&page)
{
}

String PageDebuggable::name() const
{
    String name;
    callOnMainThreadAndWait([this, protectedThis = Ref { *this }, &name] {
        RefPtr page = m_page.get();
        if (!page)
            return;

        RefPtr localTopDocument = page->localTopDocument();
        if (!localTopDocument)
            return;

        name = localTopDocument->title().isolatedCopy();
    });
    return name;
}

String PageDebuggable::url() const
{
    String url;
    callOnMainThreadAndWait([this, protectedThis = Ref { *this }, &url] {
        RefPtr page = m_page.get();
        if (!page)
            return;

        url = page->mainFrameURL().string().isolatedCopy();
        if (url.isEmpty())
            url = "about:blank"_s;
    });
    return url;
}

bool PageDebuggable::hasLocalDebugger() const
{
    bool hasLocalDebugger = false;
    callOnMainThreadAndWait([this, protectedThis = Ref { *this }, &hasLocalDebugger] {
        if (RefPtr page = m_page.get())
            hasLocalDebugger = page->protectedInspectorController()->hasLocalFrontend();
    });
    return hasLocalDebugger;
}

void PageDebuggable::connect(FrontendChannel& channel, bool isAutomaticConnection, bool immediatelyPause)
{
    callOnMainThreadAndWait([this, protectedThis = Ref { *this }, &channel, isAutomaticConnection, immediatelyPause] {
        if (RefPtr page = m_page.get())
            page->protectedInspectorController()->connectFrontend(channel, isAutomaticConnection, immediatelyPause);
    });
}

void PageDebuggable::disconnect(FrontendChannel& channel)
{
    callOnMainThreadAndWait([this, protectedThis = Ref { *this }, &channel] {
        if (RefPtr page = m_page.get())
            page->protectedInspectorController()->disconnectFrontend(channel);
    });
}

void PageDebuggable::dispatchMessageFromRemote(String&& message)
{
    callOnMainThreadAndWait([this, protectedThis = Ref { *this }, message = WTFMove(message).isolatedCopy()]() mutable {
        if (RefPtr page = m_page.get())
            page->protectedInspectorController()->dispatchMessageFromFrontend(WTFMove(message));
    });
}

void PageDebuggable::setIndicating(bool indicating)
{
    callOnMainThreadAndWait([this, protectedThis = Ref { *this }, indicating] {
        if (RefPtr page = m_page.get())
            page->protectedInspectorController()->setIndicating(indicating);
    });
}

void PageDebuggable::setNameOverride(const String& name)
{
    m_nameOverride = name;
    update();
}

void PageDebuggable::detachFromPage()
{
    m_page = nullptr;
}

} // namespace WebCore

#endif // ENABLE(REMOTE_INSPECTOR)

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 1, 2021.
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
#include "WebPageDebuggable.h"

#if ENABLE(REMOTE_INSPECTOR)

#include "WebFrameProxy.h"
#include "WebPageInspectorController.h"
#include "WebPageProxy.h"
#include <wtf/MainThread.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebPageDebuggable);

Ref<WebPageDebuggable> WebPageDebuggable::create(WebPageProxy& page)
{
    return adoptRef(*new WebPageDebuggable(page));
}

WebPageDebuggable::WebPageDebuggable(WebPageProxy& page)
    : m_page(page)
{
}

void WebPageDebuggable::detachFromPage()
{
    m_page = nullptr;
}

WebPageDebuggable::~WebPageDebuggable() = default;

String WebPageDebuggable::name() const
{
    String name;
    callOnMainRunLoopAndWait([this, protectedThis = Ref { *this }, &name] {
        RefPtr page = m_page.get();
        if (!page || !page->mainFrame())
            return;
        name = page->mainFrame()->title().isolatedCopy();
    });
    return name;
}


String WebPageDebuggable::url() const
{
    String url;
    callOnMainRunLoopAndWait([this, protectedThis = Ref { *this }, &url] {
        RefPtr page = m_page.get();
        if (!page || !page->mainFrame()) {
            url = aboutBlankURL().string().isolatedCopy();
            return;
        }

        url = page->mainFrame()->url().string().isolatedCopy();
        if (url.isEmpty())
            url = aboutBlankURL().string().isolatedCopy();
    });
    return url;
}

bool WebPageDebuggable::hasLocalDebugger() const
{
    bool hasLocalDebugger;
    callOnMainRunLoopAndWait([this, protectedThis = Ref { *this }, &hasLocalDebugger] {
        RefPtr page = m_page.get();
        hasLocalDebugger = page && page->inspectorController().hasLocalFrontend();
    });
    return hasLocalDebugger;
}

void WebPageDebuggable::connect(FrontendChannel& channel, bool isAutomaticConnection, bool immediatelyPause)
{
    callOnMainRunLoopAndWait([this, protectedThis = Ref { *this }, &channel, isAutomaticConnection, immediatelyPause] {
        if (RefPtr page = m_page.get())
            page->inspectorController().connectFrontend(channel, isAutomaticConnection, immediatelyPause);
    });
}

void WebPageDebuggable::disconnect(FrontendChannel& channel)
{
    callOnMainRunLoopAndWait([this, protectedThis = Ref { *this }, &channel] {
        if (RefPtr page = m_page.get())
            page->inspectorController().disconnectFrontend(channel);
    });
}

void WebPageDebuggable::dispatchMessageFromRemote(String&& message)
{
    callOnMainRunLoopAndWait([this, protectedThis = Ref { *this }, message = WTFMove(message).isolatedCopy()]() mutable {
        if (RefPtr page = m_page.get())
            page->inspectorController().dispatchMessageFromFrontend(WTFMove(message));
    });
}

void WebPageDebuggable::setIndicating(bool indicating)
{
    callOnMainRunLoopAndWait([this, protectedThis = Ref { *this }, indicating] {
        if (RefPtr page = m_page.get())
            page->inspectorController().setIndicating(indicating);
    });
}

void WebPageDebuggable::setNameOverride(const String& name)
{
    m_nameOverride = name;
    update();
}

} // namespace WebKit

#endif // ENABLE(REMOTE_INSPECTOR)

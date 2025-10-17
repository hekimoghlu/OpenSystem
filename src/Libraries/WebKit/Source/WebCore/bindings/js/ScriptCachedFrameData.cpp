/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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
#include "ScriptCachedFrameData.h"

#include "CommonVM.h"
#include "Document.h"
#include "GCController.h"
#include "JSDOMWindow.h"
#include "LocalFrame.h"
#include "Page.h"
#include "PageConsoleClient.h"
#include "PageGroup.h"
#include "ScriptController.h"
#include <JavaScriptCore/JSLock.h>
#include <JavaScriptCore/StrongInlines.h>
#include <JavaScriptCore/WeakGCMapInlines.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
using namespace JSC;

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScriptCachedFrameData);

ScriptCachedFrameData::ScriptCachedFrameData(LocalFrame& frame)
{
    JSLockHolder lock(commonVM());

    for (auto windowProxy : frame.windowProxy().jsWindowProxiesAsVector()) {
        auto* window = jsCast<JSDOMWindow*>(windowProxy->window());
        m_windows.add(&windowProxy->world(), Strong<JSDOMWindow>(window->vm(), window));
        window->setConsoleClient(nullptr);
    }

    frame.windowProxy().attachDebugger(nullptr);
}

ScriptCachedFrameData::~ScriptCachedFrameData()
{
    clear();
}

void ScriptCachedFrameData::restore(LocalFrame& frame)
{
    JSLockHolder lock(commonVM());

    Page* page = frame.page();

    for (auto windowProxy : frame.windowProxy().jsWindowProxiesAsVector()) {
        auto* world = &windowProxy->world();

        if (auto* window = m_windows.get(world).get())
            windowProxy->setWindow(window->vm(), *window);
        else {
            ASSERT(frame.document()->domWindow());
            auto& domWindow = *frame.document()->domWindow();
            if (&windowProxy->wrapped() == &domWindow)
                continue;

            windowProxy->setWindow(domWindow);

            if (page) {
                windowProxy->attachDebugger(page->debugger());
                windowProxy->window()->setProfileGroup(page->group().identifier());
            }
        }

        if (page)
            windowProxy->window()->setConsoleClient(page->console());
    }
}

void ScriptCachedFrameData::clear()
{
    if (m_windows.isEmpty())
        return;

    JSLockHolder lock(commonVM());
    m_windows.clear();
    GCController::singleton().garbageCollectSoon();
}

} // namespace WebCore

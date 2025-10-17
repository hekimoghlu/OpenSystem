/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 30, 2024.
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
#include "PageDebugger.h"

#include "CommonVM.h"
#include "Document.h"
#include "InspectorController.h"
#include "InspectorFrontendClient.h"
#include "JSDOMExceptionHandling.h"
#include "JSDOMWindowCustom.h"
#include "LocalFrame.h"
#include "LocalFrameView.h"
#include "Page.h"
#include "PageGroup.h"
#include "ScriptController.h"
#include "Timer.h"
#include <JavaScriptCore/JSLock.h>
#include <wtf/MainThread.h>
#include <wtf/RunLoop.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(IOS_FAMILY)
#include "WebCoreThreadInternal.h"
#endif


namespace WebCore {
using namespace JSC;
using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(PageDebugger);

PageDebugger::PageDebugger(Page& page)
    : Debugger(WebCore::commonVM())
    , m_page(page)
{
}

void PageDebugger::attachDebugger()
{
    JSC::Debugger::attachDebugger();

    m_page->setDebugger(this);
}

void PageDebugger::detachDebugger(bool isBeingDestroyed)
{
    JSC::Debugger::detachDebugger(isBeingDestroyed);

    m_page->setDebugger(nullptr);
    if (!isBeingDestroyed)
        recompileAllJSFunctions();
}

void PageDebugger::recompileAllJSFunctions()
{
    JSLockHolder lock(vm());
    Debugger::recompileAllJSFunctions();
}

void PageDebugger::didPause(JSGlobalObject* globalObject)
{
    JSC::Debugger::didPause(globalObject);

    setJavaScriptPaused(m_page->group(), true);
}

void PageDebugger::didContinue(JSGlobalObject* globalObject)
{
    JSC::Debugger::didContinue(globalObject);

    setJavaScriptPaused(m_page->group(), false);
}

void PageDebugger::runEventLoopWhilePaused()
{
    JSC::Debugger::runEventLoopWhilePaused();

#if PLATFORM(IOS_FAMILY)
    // On iOS, running an EventLoop causes us to run a nested WebRunLoop.
    // Since the WebThread is autoreleased at the end of run loop iterations
    // we need to gracefully handle releasing and reacquiring the lock.
    if (WebThreadIsEnabled()) {
        ASSERT(WebThreadIsLockedOrDisabled());
        JSC::JSLock::DropAllLocks dropAllLocks(vm());
        WebRunLoopEnableNested();

        runEventLoopWhilePausedInternal();

        WebRunLoopDisableNested();
        ASSERT(WebThreadIsLockedOrDisabled());
        return;
    }
#endif

    runEventLoopWhilePausedInternal();
}

void PageDebugger::runEventLoopWhilePausedInternal()
{
    TimerBase::fireTimersInNestedEventLoop();

    // Protect the page during the execution of the nested run loop.
    Ref protectedPage = m_page.get();

    while (!m_doneProcessingDebuggerEvents) {
        if (!platformShouldContinueRunningEventLoopWhilePaused())
            break;
    }
}

bool PageDebugger::isContentScript(JSGlobalObject* state) const
{
    return &currentWorld(*state) != &mainThreadNormalWorld() || JSC::Debugger::isContentScript(state);
}

void PageDebugger::reportException(JSGlobalObject* state, JSC::Exception* exception) const
{
    JSC::Debugger::reportException(state, exception);

    WebCore::reportException(state, exception);
}

void PageDebugger::setJavaScriptPaused(const PageGroup& pageGroup, bool paused)
{
    for (auto& page : pageGroup.pages()) {
        for (Frame* frame = &page.mainFrame(); frame; frame = frame->tree().traverseNext()) {
            auto* localFrame = dynamicDowncast<LocalFrame>(frame);
            if (!localFrame)
                continue;
            setJavaScriptPaused(*localFrame, paused);
        }

        if (auto* frontendClient = page.inspectorController().inspectorFrontendClient()) {
            if (paused)
                frontendClient->pagePaused();
            else
                frontendClient->pageUnpaused();
        }
    }
}

void PageDebugger::setJavaScriptPaused(LocalFrame& frame, bool paused)
{
    if (!frame.script().canExecuteScripts(ReasonForCallingCanExecuteScripts::NotAboutToExecuteScript))
        return;

    frame.script().setPaused(paused);

    ASSERT(frame.document());
    auto& document = *frame.document();
    if (paused) {
        document.suspendScriptedAnimationControllerCallbacks();
        document.suspendActiveDOMObjects(ReasonForSuspension::JavaScriptDebuggerPaused);
    } else {
        document.resumeActiveDOMObjects(ReasonForSuspension::JavaScriptDebuggerPaused);
        document.resumeScriptedAnimationControllerCallbacks();
    }
}

#if !PLATFORM(MAC)
bool PageDebugger::platformShouldContinueRunningEventLoopWhilePaused()
{
    return RunLoop::cycle() != RunLoop::CycleResult::Stop;
}
#endif // !PLATFORM(MAC)

} // namespace WebCore

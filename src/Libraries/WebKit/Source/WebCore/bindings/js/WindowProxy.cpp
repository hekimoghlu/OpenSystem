/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 25, 2022.
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
#include "WindowProxy.h"

#include "CommonVM.h"
#include "DOMWrapperWorld.h"
#include "GCController.h"
#include "JSDOMWindowBase.h"
#include "JSWindowProxy.h"
#include "LocalFrame.h"
#include "Page.h"
#include "PageConsoleClient.h"
#include "PageGroup.h"
#include "RemoteFrame.h"
#include "ScriptController.h"
#include "runtime_root.h"
#include <JavaScriptCore/JSLock.h>
#include <JavaScriptCore/StrongInlines.h>
#include <JavaScriptCore/WeakGCMapInlines.h>
#include <wtf/MemoryPressureHandler.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace JSC;

static void collectGarbageAfterWindowProxyDestruction()
{
    // Make sure to GC Extra Soon(tm) during memory pressure conditions
    // to soften high peaks of memory usage during navigation.
    if (MemoryPressureHandler::singleton().isUnderMemoryPressure()) {
        // NOTE: We do the collection on next runloop to ensure that there's no pointer
        //       to the window object on the stack.
        GCController::singleton().garbageCollectOnNextRunLoop();
    } else
        GCController::singleton().garbageCollectSoon();
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(WindowProxy);

WindowProxy::WindowProxy(Frame& frame)
    : m_frame(frame)
    , m_jsWindowProxies(makeUniqueRef<ProxyMap>())
{
}

WindowProxy::~WindowProxy()
{
    ASSERT(!m_frame);
    ASSERT(m_jsWindowProxies->isEmpty());
}

Frame* WindowProxy::frame() const
{
    return m_frame.get();
}

void WindowProxy::detachFromFrame()
{
    ASSERT(m_frame);

    m_frame = nullptr;

    // It's likely that destroying windowProxies will create a lot of garbage.
    if (!m_jsWindowProxies->isEmpty()) {
        while (!m_jsWindowProxies->isEmpty()) {
            auto it = m_jsWindowProxies->begin();
            it->value->window()->setConsoleClient(nullptr);
            destroyJSWindowProxy(*it->key);
        }
        collectGarbageAfterWindowProxyDestruction();
    }
}

void WindowProxy::replaceFrame(Frame& frame)
{
    ASSERT(m_frame);
    ASSERT(is<LocalFrame>(m_frame) != is<LocalFrame>(frame));
    m_frame = frame;
    setDOMWindow(frame.window());
}

void WindowProxy::destroyJSWindowProxy(DOMWrapperWorld& world)
{
    ASSERT(m_jsWindowProxies->contains(&world));
    m_jsWindowProxies->remove(&world);
    world.didDestroyWindowProxy(this);
}

JSWindowProxy& WindowProxy::createJSWindowProxy(DOMWrapperWorld& world)
{
    ASSERT(m_frame);

    ASSERT(!m_jsWindowProxies->contains(&world));
    ASSERT(m_frame->window());

    VM& vm = world.vm();

    Strong<JSWindowProxy> jsWindowProxy(vm, &JSWindowProxy::create(vm, *m_frame->window(), world));
    m_jsWindowProxies->add(&world, jsWindowProxy);
    world.didCreateWindowProxy(this);
    return *jsWindowProxy.get();
}

Vector<JSC::Strong<JSWindowProxy>> WindowProxy::jsWindowProxiesAsVector() const
{
    return copyToVector(m_jsWindowProxies->values());
}

JSDOMGlobalObject* WindowProxy::globalObject(DOMWrapperWorld& world)
{
    if (auto* windowProxy = jsWindowProxy(world))
        return windowProxy->window();
    return nullptr;
}

JSWindowProxy& WindowProxy::createJSWindowProxyWithInitializedScript(DOMWrapperWorld& world)
{
    ASSERT(m_frame);

    JSLockHolder lock(world.vm());
    auto& windowProxy = createJSWindowProxy(world);
    if (auto* localFrame = dynamicDowncast<LocalFrame>(*m_frame))
        localFrame->script().initScriptForWindowProxy(windowProxy);
    return windowProxy;
}

void WindowProxy::clearJSWindowProxiesNotMatchingDOMWindow(DOMWindow* newDOMWindow, bool goingIntoBackForwardCache)
{
    if (m_jsWindowProxies->isEmpty())
        return;

    JSLockHolder lock(commonVM());

    for (auto& windowProxy : jsWindowProxiesAsVector()) {
        if (&windowProxy->wrapped() == newDOMWindow)
            continue;

        // Clear the debugger and console from the current window before setting the new window.
        windowProxy->attachDebugger(nullptr);
        windowProxy->window()->setConsoleClient(nullptr);
        if (auto* jsDOMWindow = jsDynamicCast<JSDOMWindowBase*>(windowProxy->window()))
            jsDOMWindow->willRemoveFromWindowProxy();
    }

    // It's likely that resetting our windows created a lot of garbage, unless
    // it went in a back/forward cache.
    if (!goingIntoBackForwardCache)
        collectGarbageAfterWindowProxyDestruction();
}

void WindowProxy::setDOMWindow(DOMWindow* newDOMWindow)
{
    ASSERT(newDOMWindow);

    if (m_jsWindowProxies->isEmpty())
        return;

    ASSERT(m_frame);

    JSLockHolder lock(commonVM());

    for (auto& windowProxy : jsWindowProxiesAsVector()) {
        if (&windowProxy->wrapped() == newDOMWindow)
            continue;

        windowProxy->setWindow(*newDOMWindow);

        ScriptController* scriptController = nullptr;
        Page* page = m_frame->page();
        if (auto* localFrame = dynamicDowncast<LocalFrame>(*m_frame))
            scriptController = &localFrame->script();

        // ScriptController's m_cacheableBindingRootObject persists between page navigations
        // so needs to know about the new JSDOMWindow.
        if (auto* cacheableBindingRootObject = scriptController ? scriptController->existingCacheableBindingRootObject() : nullptr)
            cacheableBindingRootObject->updateGlobalObject(windowProxy->window());

        windowProxy->attachDebugger(page ? page->debugger() : nullptr);
        if (page) {
            windowProxy->window()->setProfileGroup(page->group().identifier());
            windowProxy->window()->setConsoleClient(page->console());
        }
    }
}

void WindowProxy::attachDebugger(JSC::Debugger* debugger)
{
    for (auto& windowProxy : m_jsWindowProxies->values())
        windowProxy->attachDebugger(debugger);
}

DOMWindow* WindowProxy::window() const
{
    return m_frame ? m_frame->window() : nullptr;
}

WindowProxy::ProxyMap WindowProxy::releaseJSWindowProxies()
{
    return std::exchange(m_jsWindowProxies, makeUniqueRef<ProxyMap>());
}

void WindowProxy::setJSWindowProxies(ProxyMap&& windowProxies)
{
    m_jsWindowProxies = makeUniqueRef<ProxyMap>(WTFMove(windowProxies));
}

} // namespace WebCore

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 30, 2022.
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
#pragma once

#include <JavaScriptCore/Strong.h>
#include <wtf/HashMap.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>
#include <wtf/WeakPtr.h>

namespace JSC {
class Debugger;
}

namespace WebCore {

class DOMWindow;
class DOMWrapperWorld;
class Frame;
class JSDOMGlobalObject;
class JSWindowProxy;

class WindowProxy : public RefCounted<WindowProxy> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(WindowProxy, WEBCORE_EXPORT);
public:
    using ProxyMap = UncheckedKeyHashMap<RefPtr<DOMWrapperWorld>, JSC::Strong<JSWindowProxy>>;

    static Ref<WindowProxy> create(Frame& frame)
    {
        return adoptRef(*new WindowProxy(frame));
    }

    WEBCORE_EXPORT ~WindowProxy();

    WEBCORE_EXPORT Frame* frame() const;
    void detachFromFrame();
    void replaceFrame(Frame&);

    void destroyJSWindowProxy(DOMWrapperWorld&);

    Vector<JSC::Strong<JSWindowProxy>> jsWindowProxiesAsVector() const;

    WEBCORE_EXPORT ProxyMap releaseJSWindowProxies();
    WEBCORE_EXPORT void setJSWindowProxies(ProxyMap&&);

    JSWindowProxy* jsWindowProxy(DOMWrapperWorld& world)
    {
        if (!m_frame)
            return nullptr;

        if (auto* existingProxy = existingJSWindowProxy(world))
            return existingProxy;

        return &createJSWindowProxyWithInitializedScript(world);
    }

    JSWindowProxy* existingJSWindowProxy(DOMWrapperWorld& world) const
    {
        auto it = m_jsWindowProxies->find(&world);
        return (it != m_jsWindowProxies->end()) ? it->value.get() : nullptr;
    }

    WEBCORE_EXPORT JSDOMGlobalObject* globalObject(DOMWrapperWorld&);

    void clearJSWindowProxiesNotMatchingDOMWindow(DOMWindow*, bool goingIntoBackForwardCache);

    WEBCORE_EXPORT void setDOMWindow(DOMWindow*);

    // Debugger can be nullptr to detach any existing Debugger.
    void attachDebugger(JSC::Debugger*); // Attaches/detaches in all worlds/window proxies.

    WEBCORE_EXPORT DOMWindow* window() const;

private:
    explicit WindowProxy(Frame&);

    JSWindowProxy& createJSWindowProxy(DOMWrapperWorld&);
    WEBCORE_EXPORT JSWindowProxy& createJSWindowProxyWithInitializedScript(DOMWrapperWorld&);

    WeakPtr<Frame> m_frame;
    UniqueRef<ProxyMap> m_jsWindowProxies;
};

} // namespace WebCore

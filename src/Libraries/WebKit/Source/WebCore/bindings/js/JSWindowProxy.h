/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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

#include "JSDOMConvertInterface.h"
#include "WindowProxy.h"
#include <JavaScriptCore/JSGlobalProxy.h>

namespace JSC {
class Debugger;
}

namespace WebCore {

class DOMWindow;
class Frame;

class WEBCORE_EXPORT JSWindowProxy final : public JSC::JSGlobalProxy {
public:
    using Base = JSC::JSGlobalProxy;
    static constexpr JSC::DestructionMode needsDestruction = JSC::NeedsDestruction;
    static void destroy(JSCell*);

    template<typename CellType, JSC::SubspaceAccess> static JSC::GCClient::IsoSubspace* subspaceFor(JSC::VM& vm) { return subspaceForImpl(vm); }

    static JSWindowProxy& create(JSC::VM&, DOMWindow&, DOMWrapperWorld&);

    DECLARE_INFO;

    JSDOMGlobalObject* window() const { return static_cast<JSDOMGlobalObject*>(target()); }

    void setWindow(JSC::VM&, JSDOMGlobalObject&);
    void setWindow(DOMWindow&);

    WindowProxy* windowProxy() const;

    DOMWindow& wrapped() const;
    static WindowProxy* toWrapped(JSC::VM&, JSC::JSValue);

    DOMWrapperWorld& world();
    Ref<DOMWrapperWorld> protectedWorld();

    void attachDebugger(JSC::Debugger*);

private:
    JSWindowProxy() = delete;
    JSWindowProxy(const JSWindowProxy&) = delete;
    JSWindowProxy(JSWindowProxy&&) = delete;
    JSWindowProxy(JSC::VM&, JSC::Structure&, DOMWrapperWorld&);
    ~JSWindowProxy();
    void finishCreation(JSC::VM&, DOMWindow&);
    static JSC::GCClient::IsoSubspace* subspaceForImpl(JSC::VM&);

#if ENABLE(WINDOW_PROXY_PROPERTY_ACCESS_NOTIFICATION)
    static bool getOwnPropertySlot(JSC::JSObject*, JSC::JSGlobalObject*, JSC::PropertyName, JSC::PropertySlot&);
    static bool getOwnPropertySlotByIndex(JSC::JSObject*, JSC::JSGlobalObject*, unsigned, JSC::PropertySlot&);
    static bool put(JSC::JSCell*, JSC::JSGlobalObject*, JSC::PropertyName, JSC::JSValue, JSC::PutPropertySlot&);
    static bool putByIndex(JSC::JSCell*, JSC::JSGlobalObject*, unsigned, JSC::JSValue, bool shouldThrow);
    static bool deleteProperty(JSC::JSCell*, JSC::JSGlobalObject*, JSC::PropertyName, JSC::DeletePropertySlot&);
    static bool deletePropertyByIndex(JSC::JSCell*, JSC::JSGlobalObject*, unsigned);
    static bool defineOwnProperty(JSC::JSObject*, JSC::JSGlobalObject*, JSC::PropertyName, const JSC::PropertyDescriptor&, bool shouldThrow);
#endif

    Ref<DOMWrapperWorld> m_world;
};

// JSWindowProxy is a little odd in that it's not a traditional wrapper and has no back pointer.
// It is, however, strongly owned by Frame via its WindowProxy, so we can get one from a WindowProxy.
WEBCORE_EXPORT JSC::JSValue toJS(JSC::JSGlobalObject*, WindowProxy&);
inline JSC::JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, WindowProxy* windowProxy) { return windowProxy ? toJS(lexicalGlobalObject, *windowProxy) : JSC::jsNull(); }
inline JSC::JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject*, WindowProxy& windowProxy) { return toJS(lexicalGlobalObject, windowProxy); }
inline JSC::JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, WindowProxy* windowProxy) { return windowProxy ? toJS(lexicalGlobalObject, globalObject, *windowProxy) : JSC::jsNull(); }

JSWindowProxy* toJSWindowProxy(WindowProxy&, DOMWrapperWorld&);
inline JSWindowProxy* toJSWindowProxy(WindowProxy* windowProxy, DOMWrapperWorld& world) { return windowProxy ? toJSWindowProxy(*windowProxy, world) : nullptr; }


template<> struct JSDOMWrapperConverterTraits<WindowProxy> {
    using WrapperClass = JSWindowProxy;
    using ToWrappedReturnType = WindowProxy*;
};

} // namespace WebCore

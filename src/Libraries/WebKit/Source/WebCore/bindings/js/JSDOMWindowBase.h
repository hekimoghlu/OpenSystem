/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 3, 2022.
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

#include "JSDOMGlobalObject.h"
#include "JSDOMWrapperCache.h"
#include "LocalDOMWindow.h"
#include <JavaScriptCore/AuxiliaryBarrierInlines.h>
#include <JavaScriptCore/HeapInlines.h>
#include <JavaScriptCore/JSArray.h>
#include <JavaScriptCore/JSCJSValueInlines.h>
#include <JavaScriptCore/JSCellInlines.h>
#include <JavaScriptCore/JSObjectInlines.h>
#include <JavaScriptCore/Lookup.h>
#include <JavaScriptCore/ObjectConstructor.h>
#include <JavaScriptCore/SlotVisitorInlines.h>
#include <JavaScriptCore/Structure.h>
#include <JavaScriptCore/StructureInlines.h>
#include <JavaScriptCore/WriteBarrier.h>
#include <cstddef>
#include <wtf/Forward.h>
#include <wtf/GetPtr.h>
#include <wtf/Vector.h>

namespace WebCore {

class DOMWrapperWorld;
class FetchResponse;
class JSDOMWindowBasePrivate;
class JSDOMWindow;
class JSWindowProxy;
class LocalFrame;

class WEBCORE_EXPORT JSDOMWindowBase : public JSDOMGlobalObject {
public:
    using Base = JSDOMGlobalObject;

    static void destroy(JSCell*);

    template<typename, JSC::SubspaceAccess>
    static void subspaceFor(JSC::VM&) { RELEASE_ASSERT_NOT_REACHED(); }

    ~JSDOMWindowBase();
    void updateDocument();

    DOMWindow& wrapped() const;
    Document* scriptExecutionContext() const;

    // Called just before removing this window from the JSWindowProxy.
    void willRemoveFromWindowProxy();

    DECLARE_INFO;

    static JSC::Structure* createStructure(JSC::VM& vm, JSC::JSValue prototype)
    {
        return JSC::Structure::create(vm, 0, prototype, JSC::TypeInfo(JSC::GlobalObjectType, StructureFlags), info());
    }

    static bool supportsRichSourceInfo(const JSC::JSGlobalObject*);
    static bool shouldInterruptScript(const JSC::JSGlobalObject*);
    static bool shouldInterruptScriptBeforeTimeout(const JSC::JSGlobalObject*);
    static JSC::RuntimeFlags javaScriptRuntimeFlags(const JSC::JSGlobalObject*);
    static void queueMicrotaskToEventLoop(JSC::JSGlobalObject&, JSC::QueuedTask&&);
    static JSC::JSObject* currentScriptExecutionOwner(JSC::JSGlobalObject*);
    static JSC::ScriptExecutionStatus scriptExecutionStatus(JSC::JSGlobalObject*, JSC::JSObject*);
    static void reportViolationForUnsafeEval(JSC::JSGlobalObject*, const String&);

    void printErrorMessage(const String&) const;

    JSWindowProxy& proxy() const;

    static void fireFrameClearedWatchpointsForWindow(LocalDOMWindow*);

    void setCurrentEvent(Event*);
    Event* currentEvent() const;

protected:
    JSDOMWindowBase(JSC::VM&, JSC::Structure*, RefPtr<DOMWindow>&&, JSWindowProxy*);
    void finishCreation(JSC::VM&, JSWindowProxy*);
    void initStaticGlobals(JSC::VM&);

    RefPtr<JSC::WatchpointSet> m_windowCloseWatchpoints;

    static const JSC::GlobalObjectMethodTable* globalObjectMethodTable();

private:
    RefPtr<DOMWindow> m_wrapped;
    RefPtr<Event> m_currentEvent;
};

WEBCORE_EXPORT JSC::JSValue toJS(JSC::JSGlobalObject*, DOMWindow&);
// The following return a JSWindowProxy or jsNull()
// JSDOMGlobalObject* is ignored, accessing a window in any context will use that LocalDOMWindow's prototype chain.
inline JSC::JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject*, DOMWindow& window) { return toJS(lexicalGlobalObject, window); }
inline JSC::JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, DOMWindow* window) { return window ? toJS(lexicalGlobalObject, globalObject, *window) : JSC::jsNull(); }
inline JSC::JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, DOMWindow* window) { return window ? toJS(lexicalGlobalObject, *window) : JSC::jsNull(); }

// The following return a JSDOMWindow or nullptr.
JSDOMWindow* toJSDOMWindow(LocalFrame&, DOMWrapperWorld&);
inline JSDOMWindow* toJSDOMWindow(LocalFrame* frame, DOMWrapperWorld& world) { return frame ? toJSDOMWindow(*frame, world) : nullptr; }

// LocalDOMWindow associated with global object of the "most-recently-entered author function or script
// on the stack, or the author function or script that originally scheduled the currently-running callback."
// (<https://html.spec.whatwg.org/multipage/webappapis.html#concept-incumbent-everything>, 27 April 2017)
// FIXME: Make this work for an "author function or script that originally scheduled the currently-running callback."
// See <https://bugs.webkit.org/show_bug.cgi?id=163412>.
LocalDOMWindow& incumbentDOMWindow(JSC::JSGlobalObject&, JSC::CallFrame&);
LocalDOMWindow& incumbentDOMWindow(JSC::JSGlobalObject&);

LocalDOMWindow& activeDOMWindow(JSC::JSGlobalObject&);
LocalDOMWindow& firstDOMWindow(JSC::JSGlobalObject&);

LocalDOMWindow& legacyActiveDOMWindowForAccessor(JSC::JSGlobalObject&, JSC::CallFrame&);
LocalDOMWindow& legacyActiveDOMWindowForAccessor(JSC::JSGlobalObject&);

} // namespace WebCore

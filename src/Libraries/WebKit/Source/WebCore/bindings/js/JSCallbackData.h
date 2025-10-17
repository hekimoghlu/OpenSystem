/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 26, 2024.
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

#include "JSDOMBinding.h"
#include "ScriptExecutionContext.h"
#include <JavaScriptCore/JSObject.h>
#include <JavaScriptCore/Weak.h>
#include <JavaScriptCore/WeakInlines.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Threading.h>

namespace WebCore {

template<typename ImplementationClass> struct JSDOMCallbackConverterTraits;

// We have to clean up this data on the context thread because unprotecting a
// JSObject on the wrong thread without synchronization would corrupt the heap
// (and synchronization would be slow).

class JSCallbackData {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(JSCallbackData, WEBCORE_EXPORT);
public:
    enum class CallbackType { Function, Object, FunctionOrObject };

    JSCallbackData(JSC::JSObject* callback, JSDOMGlobalObject* globalObject, void* owner)
        : m_globalObject(globalObject)
        , m_callback(callback, &m_weakOwner, owner)
    {
    }

    ~JSCallbackData()
    {
#if !PLATFORM(IOS_FAMILY)
        ASSERT(m_thread.ptr() == &Thread::current());
#endif
    }

    JSDOMGlobalObject* globalObject() { return m_globalObject.get(); }
    JSC::JSObject* callback() { return m_callback.get(); }

    template<typename Visitor> void visitJSFunction(Visitor&);

    WEBCORE_EXPORT static JSC::JSValue invokeCallback(JSDOMGlobalObject&, JSC::JSObject* callback, JSC::JSValue thisValue, JSC::MarkedArgumentBuffer&, CallbackType, JSC::PropertyName functionName, NakedPtr<JSC::Exception>& returnedException);

    JSC::JSValue invokeCallback(JSC::JSValue thisValue, JSC::MarkedArgumentBuffer& args, CallbackType callbackType, JSC::PropertyName functionName, NakedPtr<JSC::Exception>& returnedException)
    {
        auto* globalObject = this->globalObject();
        if (!globalObject)
            return { };

        return JSCallbackData::invokeCallback(*globalObject, callback(), thisValue, args, callbackType, functionName, returnedException);
    }

private:
    JSC::Weak<JSDOMGlobalObject> m_globalObject;

    class WeakOwner : public JSC::WeakHandleOwner {
        bool isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown>, void* owner, JSC::AbstractSlotVisitor& visitor, ASCIILiteral* reason) override
        {
            if (UNLIKELY(reason))
                *reason = "Callback owner is an opaque root"_s;
            return visitor.containsOpaqueRoot(owner);
        }
    };
    WeakOwner m_weakOwner;
    JSC::Weak<JSC::JSObject> m_callback;

#if ASSERT_ENABLED
    Ref<Thread> m_thread { Thread::current() };
#endif
};

class DeleteCallbackDataTask : public ScriptExecutionContext::Task {
public:
    template <typename CallbackDataType>
    explicit DeleteCallbackDataTask(CallbackDataType* data)
        : ScriptExecutionContext::Task(ScriptExecutionContext::Task::CleanupTask, [data = std::unique_ptr<CallbackDataType>(data)] (ScriptExecutionContext&) {
        })
    {
    }
};

} // namespace WebCore

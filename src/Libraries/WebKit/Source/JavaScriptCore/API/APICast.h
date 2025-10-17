/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 5, 2022.
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

#include <wtf/Compiler.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

#include "Integrity.h"
#include "JSAPIValueWrapper.h"
#include "JSCJSValue.h"
#include "JSCJSValueInlines.h"
#include "HeapCellInlines.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

namespace JSC {
    class CallFrame;
    class PropertyNameArray;
    class VM;
    class JSObject;
    class JSValue;
}

typedef const struct OpaqueJSContextGroup* JSContextGroupRef;
typedef const struct OpaqueJSContext* JSContextRef;
typedef struct OpaqueJSContext* JSGlobalContextRef;
typedef struct OpaqueJSPropertyNameAccumulator* JSPropertyNameAccumulatorRef;
typedef const struct OpaqueJSValue* JSValueRef;
typedef struct OpaqueJSValue* JSObjectRef;

/* Opaque typing convenience methods */

inline JSC::JSGlobalObject* toJS(JSContextRef context)
{
    ASSERT(context);
    return JSC::Integrity::audit(reinterpret_cast<JSC::JSGlobalObject*>(const_cast<OpaqueJSContext*>(context)));
}

inline JSC::JSGlobalObject* toJS(JSGlobalContextRef context)
{
    ASSERT(context);
    return JSC::Integrity::audit(reinterpret_cast<JSC::JSGlobalObject*>(context));
}

inline JSC::JSGlobalObject* toJSGlobalObject(JSGlobalContextRef context)
{
    return toJS(context);
}

inline JSC::JSValue toJS(JSC::JSGlobalObject* globalObject, JSValueRef v)
{
    ASSERT_UNUSED(globalObject, globalObject);
#if !CPU(ADDRESS64)
    JSC::JSCell* jsCell = reinterpret_cast<JSC::JSCell*>(const_cast<OpaqueJSValue*>(v));
    if (!jsCell)
        return JSC::jsNull();
    JSC::JSValue result;
    if (jsCell->isAPIValueWrapper())
        result = JSC::jsCast<JSC::JSAPIValueWrapper*>(jsCell)->value();
    else
        result = jsCell;
#else
    JSC::JSValue result = std::bit_cast<JSC::JSValue>(v);
#endif
    if (!result)
        return JSC::jsNull();
    if (result.isCell()) {
        JSC::Integrity::audit(result.asCell());
        RELEASE_ASSERT(result.asCell()->methodTable());
    }
    return result;
}

#if CPU(ADDRESS64)
inline JSC::JSValue toJS(JSValueRef value)
{
    return JSC::Integrity::audit(std::bit_cast<JSC::JSValue>(value));
}
#endif

inline JSC::JSValue toJSForGC(JSC::JSGlobalObject* globalObject, JSValueRef v)
{
    ASSERT_UNUSED(globalObject, globalObject);
#if !CPU(ADDRESS64)
    JSC::JSCell* jsCell = reinterpret_cast<JSC::JSCell*>(const_cast<OpaqueJSValue*>(v));
    if (!jsCell)
        return JSC::JSValue();
    JSC::JSValue result = jsCell;
#else
    JSC::JSValue result = std::bit_cast<JSC::JSValue>(v);
#endif
    if (result && result.isCell()) {
        JSC::Integrity::audit(result.asCell());
        RELEASE_ASSERT(result.asCell()->methodTable());
    }
    return result;
}

// Used in JSObjectGetPrivate as that may be called during finalization
inline JSC::JSObject* uncheckedToJS(JSObjectRef o)
{
    return JSC::Integrity::audit(reinterpret_cast<JSC::JSObject*>(o));
}

inline JSC::JSObject* toJS(JSObjectRef o)
{
    JSC::JSObject* object = uncheckedToJS(o);
    if (object)
        RELEASE_ASSERT(object->methodTable());
    return object;
}

inline JSC::PropertyNameArray* toJS(JSPropertyNameAccumulatorRef a)
{
    return reinterpret_cast<JSC::PropertyNameArray*>(a);
}

inline JSC::VM* toJS(JSContextGroupRef g)
{
    return JSC::Integrity::audit(reinterpret_cast<JSC::VM*>(const_cast<OpaqueJSContextGroup*>(g)));
}

inline JSValueRef toRef(JSC::VM& vm, JSC::JSValue v)
{
    ASSERT(vm.currentThreadIsHoldingAPILock());
#if !CPU(ADDRESS64)
    if (!v)
        return 0;
    if (!v.isCell())
        return reinterpret_cast<JSValueRef>(JSC::JSAPIValueWrapper::create(vm, v));
    return reinterpret_cast<JSValueRef>(v.asCell());
#else
    UNUSED_PARAM(vm);
    return std::bit_cast<JSValueRef>(JSC::Integrity::audit(v));
#endif
}

inline JSValueRef toRef(JSC::JSGlobalObject* globalObject, JSC::JSValue v)
{
    return toRef(getVM(globalObject), v);
}

#if CPU(ADDRESS64)
inline JSValueRef toRef(JSC::JSValue v)
{
    return std::bit_cast<JSValueRef>(JSC::Integrity::audit(v));
}
#endif

inline JSObjectRef toRef(JSC::JSObject* o)
{
    return reinterpret_cast<JSObjectRef>(JSC::Integrity::audit(o));
}

inline JSObjectRef toRef(const JSC::JSObject* o)
{
    return reinterpret_cast<JSObjectRef>(JSC::Integrity::audit(const_cast<JSC::JSObject*>(o)));
}

inline JSContextRef toRef(JSC::JSGlobalObject* globalObject)
{
    return reinterpret_cast<JSContextRef>(JSC::Integrity::audit(globalObject));
}

inline JSGlobalContextRef toGlobalRef(JSC::JSGlobalObject* globalObject)
{
    return reinterpret_cast<JSGlobalContextRef>(JSC::Integrity::audit(globalObject));
}

inline JSPropertyNameAccumulatorRef toRef(JSC::PropertyNameArray* l)
{
    return reinterpret_cast<JSPropertyNameAccumulatorRef>(l);
}

inline JSContextGroupRef toRef(JSC::VM* g)
{
    return reinterpret_cast<JSContextGroupRef>(JSC::Integrity::audit(g));
}

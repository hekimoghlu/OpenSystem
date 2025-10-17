/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 25, 2024.
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

#include "ArgList.h"
#include "Exception.h"
#include "JSCJSValue.h"
#include "JSCJSValueInlines.h"
#include "JSObject.h"
#include "Strong.h"
#include "StrongInlines.h"
#include <wtf/Expected.h>
#include <wtf/JSONValues.h>
#include <wtf/text/WTFString.h>

namespace JSC {
class JSValue;
}

namespace Inspector {

class ScriptCallArgumentHandler {
public:
    ScriptCallArgumentHandler(JSC::JSGlobalObject* globalObject) : m_globalObject(globalObject) { }

    void appendArgument(const char*);
    void appendArgument(const String&);
    void appendArgument(JSC::JSValue);
    void appendArgument(long);
    void appendArgument(long long);
    void appendArgument(unsigned int);
    void appendArgument(uint64_t);
    JS_EXPORT_PRIVATE void appendArgument(int);
    void appendArgument(bool);

protected:
    JSC::MarkedArgumentBuffer m_arguments;
    JSC::JSGlobalObject* const m_globalObject;

private:
    // MarkedArgumentBuffer must be stack allocated, so prevent heap
    // alloc of ScriptFunctionCall as well.
    void* operator new(size_t) { ASSERT_NOT_REACHED(); return reinterpret_cast<void*>(0xbadbeef); }
    void* operator new[](size_t) { ASSERT_NOT_REACHED(); return reinterpret_cast<void*>(0xbadbeef); }
};

class ScriptFunctionCall : public ScriptCallArgumentHandler {
public:
    typedef JSC::JSValue (*ScriptFunctionCallHandler)(JSC::JSGlobalObject*, JSC::JSValue functionObject, const JSC::CallData& callData, JSC::JSValue thisValue, const JSC::ArgList& args, NakedPtr<JSC::Exception>&);
    JS_EXPORT_PRIVATE ScriptFunctionCall(JSC::JSGlobalObject*, JSC::JSObject* thisObject, const String& name, ScriptFunctionCallHandler = nullptr);
    JS_EXPORT_PRIVATE Expected<JSC::JSValue, NakedPtr<JSC::Exception>> call();

protected:
    ScriptFunctionCallHandler m_callHandler;
    JSC::Strong<JSC::JSObject> m_thisObject;
    String m_name;
};

} // namespace Inspector

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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

#include "CallData.h"

namespace WTF {
class Stopwatch;
}

namespace JSC {
class Debugger;
class Exception;
class SourceCode;
class VM;
}

namespace Inspector {

typedef JSC::JSValue (*InspectorFunctionCallHandler)(JSC::JSGlobalObject* globalObject, JSC::JSValue functionObject, const JSC::CallData& callData, JSC::JSValue thisValue, const JSC::ArgList& args, NakedPtr<JSC::Exception>& returnedException);
typedef JSC::JSValue (*InspectorEvaluateHandler)(JSC::JSGlobalObject*, const JSC::SourceCode&, JSC::JSValue thisValue, NakedPtr<JSC::Exception>& returnedException);

class InspectorEnvironment {
public:
    virtual ~InspectorEnvironment() { }
    virtual bool developerExtrasEnabled() const = 0;
    virtual bool canAccessInspectedScriptState(JSC::JSGlobalObject*) const = 0;
    virtual InspectorFunctionCallHandler functionCallHandler() const = 0;
    virtual InspectorEvaluateHandler evaluateHandler() const = 0;
    virtual void frontendInitialized() = 0;
    virtual WTF::Stopwatch& executionStopwatch() const = 0;
    virtual JSC::Debugger* debugger() = 0;
    virtual JSC::VM& vm() = 0;
};

} // namespace Inspector

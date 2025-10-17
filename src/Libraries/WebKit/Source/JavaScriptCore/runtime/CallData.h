/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 20, 2022.
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

#include "NativeFunction.h"
#include <wtf/Forward.h>
#include <wtf/NakedPtr.h>

namespace JSC {

class ArgList;
class Exception;
class FunctionExecutable;
class JSScope;

struct CallData {
    enum class Type : uint8_t { None, Native, JS };
    Type type { Type::None };

    CallData() { } // Needed for the anonymous union below.
    union {
        struct {
            TaggedNativeFunction function;
            bool isBoundFunction;
            bool isWasm;
        } native;
        struct {
            FunctionExecutable* functionExecutable;
            JSScope* scope;
        } js;
    };
};

enum class ProfilingReason : uint8_t {
    API,
    Microtask,
    Other
};

// Convenience wrapper so you don't need to deal with CallData unless you are going to use it.
JS_EXPORT_PRIVATE JSValue call(JSGlobalObject*, JSValue functionObject, const ArgList&, ASCIILiteral errorMessage);
JS_EXPORT_PRIVATE JSValue call(JSGlobalObject*, JSValue functionObject, JSValue thisValue, const ArgList&, ASCIILiteral errorMessage);

JS_EXPORT_PRIVATE JSValue call(JSGlobalObject*, JSValue functionObject, const CallData&, JSValue thisValue, const ArgList&);
JS_EXPORT_PRIVATE JSValue call(JSGlobalObject*, JSValue functionObject, const CallData&, JSValue thisValue, const ArgList&, NakedPtr<Exception>& returnedException);

JS_EXPORT_PRIVATE JSValue profiledCall(JSGlobalObject*, ProfilingReason, JSValue functionObject, const CallData&, JSValue thisValue, const ArgList&);
JS_EXPORT_PRIVATE JSValue profiledCall(JSGlobalObject*, ProfilingReason, JSValue functionObject, const CallData&, JSValue thisValue, const ArgList&, NakedPtr<Exception>& returnedException);

} // namespace JSC

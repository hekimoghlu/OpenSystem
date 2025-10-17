/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 29, 2025.
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
#include "JSCTestRunnerUtils.h"

#include "APICast.h"
#include "JSGlobalObjectInlines.h"
#include "TestRunnerUtils.h"

namespace JSC {


JSValueRef failNextNewCodeBlock(JSContextRef context)
{
    JSGlobalObject* globalObject= toJS(context);
    JSLockHolder holder(globalObject);
    return toRef(globalObject, failNextNewCodeBlock(globalObject));
}

JSValueRef numberOfDFGCompiles(JSContextRef context, JSValueRef theFunctionValueRef)
{
    JSGlobalObject* globalObject= toJS(context);
    JSLockHolder holder(globalObject);
    return toRef(globalObject, numberOfDFGCompiles(toJS(globalObject, theFunctionValueRef)));
}

JSValueRef setNeverInline(JSContextRef context, JSValueRef theFunctionValueRef)
{
    JSGlobalObject* globalObject= toJS(context);
    JSLockHolder holder(globalObject);
    return toRef(globalObject, setNeverInline(toJS(globalObject, theFunctionValueRef)));
}

JSValueRef setNeverOptimize(JSContextRef context, JSValueRef theFunctionValueRef)
{
    JSGlobalObject* globalObject= toJS(context);
    JSLockHolder holder(globalObject);
    return toRef(globalObject, setNeverOptimize(toJS(globalObject, theFunctionValueRef)));
}

} // namespace JSC


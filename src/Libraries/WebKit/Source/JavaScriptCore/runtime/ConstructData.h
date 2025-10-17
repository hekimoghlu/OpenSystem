/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 1, 2021.
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

namespace JSC {

class JSObject;

// Convenience wrapper so you don't need to deal with CallData unless you are going to use it.
JS_EXPORT_PRIVATE JSObject* construct(JSGlobalObject*, JSValue functionObject, const ArgList&, ASCIILiteral errorMessage);
JS_EXPORT_PRIVATE JSObject* construct(JSGlobalObject*, JSValue functionObject, JSValue newTarget, const ArgList&, ASCIILiteral errorMessage);

JS_EXPORT_PRIVATE JSObject* construct(JSGlobalObject*, JSValue constructor, const CallData&, const ArgList&, JSValue newTarget);

ALWAYS_INLINE JSObject* construct(JSGlobalObject* globalObject, JSValue constructorObject, const CallData& callData, const ArgList& args)
{
    return construct(globalObject, constructorObject, callData, args, constructorObject);
}

JS_EXPORT_PRIVATE JSObject* profiledConstruct(JSGlobalObject*, ProfilingReason, JSValue constructor, const CallData&, const ArgList&, JSValue newTarget);

ALWAYS_INLINE JSObject* profiledConstruct(JSGlobalObject* globalObject, ProfilingReason reason, JSValue constructorObject, const CallData& callData, const ArgList& args)
{
    return profiledConstruct(globalObject, reason, constructorObject, callData, args, constructorObject);
}

} // namespace JSC

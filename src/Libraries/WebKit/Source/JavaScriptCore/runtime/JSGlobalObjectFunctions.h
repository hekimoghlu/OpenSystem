/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 6, 2023.
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

#include "JSCJSValue.h"
#include <unicode/uchar.h>

namespace JSC {

class ArgList;
class CallFrame;
class JSObject;

// FIXME: These functions should really be in JSGlobalObject.cpp, but putting them there
// is a 0.5% reduction.

extern const ASCIILiteral ObjectProtoCalledOnNullOrUndefinedError;
extern const ASCIILiteral RestrictedPropertyAccessError;

JSC_DECLARE_HOST_FUNCTION(globalFuncEval);
JSC_DECLARE_HOST_FUNCTION(globalFuncParseInt);
JSC_DECLARE_HOST_FUNCTION(globalFuncParseFloat);
JSC_DECLARE_HOST_FUNCTION(globalFuncDecodeURI);
JSC_DECLARE_HOST_FUNCTION(globalFuncDecodeURIComponent);
JSC_DECLARE_HOST_FUNCTION(globalFuncEncodeURI);
JSC_DECLARE_HOST_FUNCTION(globalFuncEncodeURIComponent);
JSC_DECLARE_HOST_FUNCTION(globalFuncEscape);
JSC_DECLARE_HOST_FUNCTION(globalFuncUnescape);
JSC_DECLARE_HOST_FUNCTION(globalFuncThrowTypeError);
JSC_DECLARE_HOST_FUNCTION(globalFuncThrowTypeErrorArgumentsCalleeAndCaller);
JSC_DECLARE_HOST_FUNCTION(globalFuncMakeTypeError);
JSC_DECLARE_HOST_FUNCTION(globalFuncProtoGetter);
JSC_DECLARE_HOST_FUNCTION(globalFuncProtoSetter);
JSC_DECLARE_HOST_FUNCTION(globalFuncSetPrototypeDirect);
JSC_DECLARE_HOST_FUNCTION(globalFuncSetPrototypeDirectOrThrow);
JSC_DECLARE_HOST_FUNCTION(globalFuncHostPromiseRejectionTracker);
JSC_DECLARE_HOST_FUNCTION(globalFuncBuiltinLog);
JSC_DECLARE_HOST_FUNCTION(globalFuncBuiltinDescribe);
JSC_DECLARE_HOST_FUNCTION(globalFuncImportModule);
JSC_DECLARE_HOST_FUNCTION(globalFuncCopyDataProperties);
JSC_DECLARE_HOST_FUNCTION(globalFuncCloneObject);
JSC_DECLARE_HOST_FUNCTION(globalFuncHandleNegativeProxyHasTrapResult);
JSC_DECLARE_HOST_FUNCTION(globalFuncHandlePositiveProxySetTrapResult);
JSC_DECLARE_HOST_FUNCTION(globalFuncHandleProxyGetTrapResult);
JSC_DECLARE_HOST_FUNCTION(globalFuncIsNaN);
JSC_DECLARE_HOST_FUNCTION(globalFuncToIntegerOrInfinity);
JSC_DECLARE_HOST_FUNCTION(globalFuncToLength);
JSC_DECLARE_HOST_FUNCTION(globalFuncSpeciesGetter);

JS_EXPORT_PRIVATE double jsToNumber(StringView);

} // namespace JSC

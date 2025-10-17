/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 24, 2024.
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

#include "JSCBuiltins.h"

namespace JSC {

class CodeBlock;
class JSGlobalObject;

#define JSC_FOREACH_LINK_TIME_CONSTANTS(v) \
    JSC_FOREACH_BUILTIN_LINK_TIME_CONSTANT(v) \
    v(throwTypeErrorFunction, nullptr) \
    v(importModule, nullptr) \
    v(mapStorage, nullptr) \
    v(mapIterationNext, nullptr) \
    v(mapIterationEntry, nullptr) \
    v(mapIterationEntryKey, nullptr) \
    v(mapIterationEntryValue, nullptr) \
    v(mapIteratorNext, nullptr) \
    v(mapIteratorKey, nullptr) \
    v(mapIteratorValue, nullptr) \
    v(setStorage, nullptr) \
    v(setIterationNext, nullptr) \
    v(setIterationEntry, nullptr) \
    v(setIterationEntryKey, nullptr) \
    v(setIteratorNext, nullptr) \
    v(setIteratorKey, nullptr) \
    v(setClone, nullptr) \
    v(setPrototypeDirect, nullptr) \
    v(setPrototypeDirectOrThrow, nullptr) \
    v(copyDataProperties, nullptr) \
    v(cloneObject, nullptr) \
    v(enqueueJob, nullptr) \
    v(makeTypeError, nullptr) \
    v(AggregateError, nullptr) \
    v(typedArrayLength, nullptr) \
    v(typedArrayContentType, nullptr) \
    v(typedArrayGetOriginalConstructor, nullptr) \
    v(toIntegerOrInfinity, nullptr) \
    v(toLength, nullptr) \
    v(isTypedArrayView, nullptr) \
    v(isSharedTypedArrayView, nullptr) \
    v(isResizableOrGrowableSharedTypedArrayView, nullptr) \
    v(typedArrayFromFast, nullptr) \
    v(isDetached, nullptr) \
    v(isBoundFunction, nullptr) \
    v(hasInstanceBoundFunction, nullptr) \
    v(instanceOf, nullptr) \
    v(BuiltinLog, nullptr) \
    v(BuiltinDescribe, nullptr) \
    v(RegExp, nullptr) \
    v(Iterator, nullptr) \
    v(min, nullptr) \
    v(Promise, nullptr) \
    v(InternalPromise, nullptr) \
    v(defaultPromiseThen, nullptr) \
    v(repeatCharacter, nullptr) \
    v(isArray, nullptr) \
    v(isArraySlow, nullptr) \
    v(appendMemcpy, nullptr) \
    v(hostPromiseRejectionTracker, nullptr) \
    v(Set, nullptr) \
    v(Map, nullptr) \
    v(importInRealm, nullptr) \
    v(evalFunction, nullptr) \
    v(evalInRealm, nullptr) \
    v(moveFunctionToRealm, nullptr) \
    v(isConstructor, nullptr) \
    v(sameValue, nullptr) \
    v(regExpProtoFlagsGetter, nullptr) \
    v(regExpProtoGlobalGetter, nullptr) \
    v(regExpProtoHasIndicesGetter, nullptr) \
    v(regExpProtoIgnoreCaseGetter, nullptr) \
    v(regExpProtoMultilineGetter, nullptr) \
    v(regExpProtoSourceGetter, nullptr) \
    v(regExpProtoStickyGetter, nullptr) \
    v(regExpProtoDotAllGetter, nullptr) \
    v(regExpProtoUnicodeGetter, nullptr) \
    v(regExpProtoUnicodeSetsGetter, nullptr) \
    v(regExpBuiltinExec, nullptr) \
    v(regExpCreate, nullptr) \
    v(isRegExp, nullptr) \
    v(regExpMatchFast, nullptr) \
    v(regExpSearchFast, nullptr) \
    v(regExpSplitFast, nullptr) \
    v(regExpPrototypeSymbolMatch, nullptr) \
    v(regExpPrototypeSymbolReplace, nullptr) \
    v(regExpTestFast, nullptr) \
    v(stringIncludesInternal, nullptr) \
    v(stringIndexOfInternal, nullptr) \
    v(stringSplitFast, nullptr) \
    v(stringSubstring, nullptr) \
    v(handleNegativeProxyHasTrapResult, nullptr) \
    v(handlePositiveProxySetTrapResult, nullptr) \
    v(handleProxyGetTrapResult, nullptr) \
    v(webAssemblyCompileStreamingInternal, nullptr) \
    v(webAssemblyInstantiateStreamingInternal, nullptr) \
    v(Object, nullptr) \
    v(Array, nullptr) \
    v(applyFunction, nullptr) \
    v(callFunction, nullptr) \
    v(hasOwnPropertyFunction, nullptr) \
    v(createPrivateSymbol, nullptr) \
    v(emptyPropertyNameEnumerator, nullptr) \
    v(sentinelString, nullptr) \
    v(createRemoteFunction, nullptr) \
    v(isRemoteFunction, nullptr) \
    v(arrayFromFastFillWithUndefined, nullptr) \
    v(arrayFromFastFillWithEmpty, nullptr) \
    v(jsonParse, nullptr) \
    v(jsonStringify, nullptr) \
    v(String, nullptr) \
    v(Int8Array, nullptr) \
    v(Uint8Array, nullptr) \
    v(Uint8ClampedArray, nullptr) \
    v(Int16Array, nullptr) \
    v(Uint16Array, nullptr) \
    v(Int32Array, nullptr) \
    v(Uint32Array, nullptr) \
    v(Float16Array, nullptr) \
    v(Float32Array, nullptr) \
    v(Float64Array, nullptr) \
    v(BigInt64Array, nullptr) \
    v(BigUint64Array, nullptr) \
    v(wrapForValidIteratorCreate, nullptr) \
    v(asyncFromSyncIteratorCreate, nullptr) \
    v(regExpStringIteratorCreate, nullptr) \
    v(iteratorHelperCreate, nullptr) \


#define DECLARE_LINK_TIME_CONSTANT(name, code) name,
enum class LinkTimeConstant : int32_t {
    JSC_FOREACH_LINK_TIME_CONSTANTS(DECLARE_LINK_TIME_CONSTANT)
};
#undef DECLARE_LINK_TIME_CONSTANT
#define COUNT_LINK_TIME_CONSTANT(name, code) 1 +
static constexpr unsigned numberOfLinkTimeConstants = JSC_FOREACH_LINK_TIME_CONSTANTS(COUNT_LINK_TIME_CONSTANT) 0;
#undef COUNT_LINK_TIME_CONSTANT

} // namespace JSC

namespace WTF {

class PrintStream;

void printInternal(PrintStream&, JSC::LinkTimeConstant);

} // namespace WTF

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 19, 2023.
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

// The purpose of this header file is simply to #include all the types
// that we have Subspaces for so that Heap.cpp's #include list is not flooded
// with these, and that it'll be easier to discern between these Subspace types
// from other data structures needed for implementing Heap.

#include "BigIntObject.h"
#include "BooleanObject.h"
#include "BrandedStructure.h"
#include "ClonedArguments.h"
#include "DateInstance.h"
#include "DebuggerScope.h"
#include "GetterSetter.h"
#include "IntlCollator.h"
#include "IntlDateTimeFormat.h"
#include "IntlDisplayNames.h"
#include "IntlDurationFormat.h"
#include "IntlListFormat.h"
#include "IntlLocale.h"
#include "IntlNumberFormat.h"
#include "IntlPluralRules.h"
#include "IntlRelativeTimeFormat.h"
#include "IntlSegmentIterator.h"
#include "IntlSegmenter.h"
#include "IntlSegments.h"
#include "JSAPIGlobalObject.h"
#include "JSAPIValueWrapper.h"
#include "JSAPIWrapperObject.h"
#include "JSArray.h"
#include "JSArrayBuffer.h"
#include "JSArrayIterator.h"
#include "JSAsyncFromSyncIterator.h"
#include "JSAsyncGenerator.h"
#include "JSBigInt.h"
#include "JSBoundFunction.h"
#include "JSCallbackConstructor.h"
#include "JSCallbackFunction.h"
#include "JSCallbackObject.h"
#include "JSCallee.h"
#include "JSCustomGetterFunction.h"
#include "JSCustomSetterFunction.h"
#include "JSDataView.h"
#include "JSFunction.h"
#include "JSGlobalLexicalEnvironment.h"
#include "JSGlobalObject.h"
#include "JSGlobalProxy.h"
#include "JSInjectedScriptHost.h"
#include "JSIteratorHelper.h"
#include "JSJavaScriptCallFrame.h"
#include "JSMap.h"
#include "JSMapIterator.h"
#include "JSModuleNamespaceObject.h"
#include "JSModuleRecord.h"
#include "JSNativeStdFunction.h"
#include "JSPromise.h"
#include "JSPropertyNameEnumerator.h"
#include "JSRegExpStringIterator.h"
#include "JSScriptFetchParameters.h"
#include "JSScriptFetcher.h"
#include "JSSet.h"
#include "JSSetIterator.h"
#include "JSSourceCode.h"
#include "JSString.h"
#include "JSStringIterator.h"
#include "JSTemplateObjectDescriptor.h"
#include "JSTypedArrays.h"
#include "JSWebAssemblyArray.h"
#include "JSWebAssemblyException.h"
#include "JSWebAssemblyGlobal.h"
#include "JSWebAssemblyInstance.h"
#include "JSWebAssemblyMemory.h"
#include "JSWebAssemblyModule.h"
#include "JSWebAssemblyStruct.h"
#include "JSWebAssemblyTable.h"
#include "JSWebAssemblyTag.h"
#include "JSWithScope.h"
#include "JSWrapForValidIterator.h"
#include "NativeExecutable.h"
#include "ProgramExecutable.h"
#include "PropertyTable.h"
#include "ProxyRevoke.h"
#include "RegExpObject.h"
#include "ScopedArguments.h"
#include "ShadowRealmObject.h"
#include "StrictEvalActivation.h"
#include "StringObject.h"
#include "StructureChain.h"
#include "SymbolObject.h"
#include "SyntheticModuleRecord.h"
#include "TemporalCalendar.h"
#include "TemporalDuration.h"
#include "TemporalInstant.h"
#include "TemporalPlainDate.h"
#include "TemporalPlainDateTime.h"
#include "TemporalPlainTime.h"
#include "TemporalTimeZone.h"
#include "UnlinkedFunctionCodeBlock.h"
#include "UnlinkedModuleProgramCodeBlock.h"
#include "UnlinkedProgramCodeBlock.h"
#include "WebAssemblyFunction.h"
#include "WebAssemblyModuleRecord.h"
#include "WebAssemblyWrapperFunction.h"

#if JSC_OBJC_API_ENABLED
#include "ObjCCallbackFunction.h"
#endif

#ifdef JSC_GLIB_API_ENABLED
#include "JSAPIWrapperGlobalObject.h"
#include "JSCCallbackFunction.h"
#endif

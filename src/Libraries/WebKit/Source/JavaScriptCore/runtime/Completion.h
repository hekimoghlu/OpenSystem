/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 23, 2022.
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
#include "Identifier.h"
#include "JSCJSValue.h"
#include "ScriptFetchParameters.h"
#include <wtf/FileSystem.h>
#include <wtf/NakedPtr.h>

namespace JSC {

class BytecodeCacheError;
class CachedBytecode;
class CallFrame;
class Exception;
class JSObject;
class ParserError;
class SourceCode;
class Symbol;
class VM;
class JSInternalPromise;

JS_EXPORT_PRIVATE bool checkSyntax(VM&, const SourceCode&, ParserError&);
JS_EXPORT_PRIVATE bool checkSyntax(JSGlobalObject*, const SourceCode&, JSValue* exception = nullptr);
JS_EXPORT_PRIVATE bool checkModuleSyntax(JSGlobalObject*, const SourceCode&, ParserError&);

JS_EXPORT_PRIVATE RefPtr<CachedBytecode> generateProgramBytecode(VM&, const SourceCode&, FileSystem::PlatformFileHandle fd, BytecodeCacheError&);
JS_EXPORT_PRIVATE RefPtr<CachedBytecode> generateModuleBytecode(VM&, const SourceCode&, FileSystem::PlatformFileHandle fd, BytecodeCacheError&);

JS_EXPORT_PRIVATE JSValue evaluate(JSGlobalObject*, const SourceCode&, JSValue thisValue, NakedPtr<Exception>& returnedException);
inline JSValue evaluate(JSGlobalObject* globalObject, const SourceCode& sourceCode, JSValue thisValue = JSValue())
{
    NakedPtr<Exception> unused;
    return evaluate(globalObject, sourceCode, thisValue, unused);
}

JS_EXPORT_PRIVATE JSValue profiledEvaluate(JSGlobalObject*, ProfilingReason, const SourceCode&, JSValue thisValue, NakedPtr<Exception>& returnedException);
inline JSValue profiledEvaluate(JSGlobalObject* globalObject, ProfilingReason reason, const SourceCode& sourceCode, JSValue thisValue = JSValue())
{
    NakedPtr<Exception> unused;
    return profiledEvaluate(globalObject, reason, sourceCode, thisValue, unused);
}

JS_EXPORT_PRIVATE JSValue evaluateWithScopeExtension(JSGlobalObject*, const SourceCode&, JSObject* scopeExtension, NakedPtr<Exception>& returnedException);

// Load the module source and evaluate it.
JS_EXPORT_PRIVATE JSInternalPromise* loadAndEvaluateModule(JSGlobalObject*, Symbol* moduleId, JSValue parameters, JSValue scriptFetcher);
JS_EXPORT_PRIVATE JSInternalPromise* loadAndEvaluateModule(JSGlobalObject*, const String& moduleName, JSValue parameters, JSValue scriptFetcher);
JS_EXPORT_PRIVATE JSInternalPromise* loadAndEvaluateModule(JSGlobalObject*, const SourceCode&, JSValue scriptFetcher);

// Fetch the module source, and instantiate the module record.
JS_EXPORT_PRIVATE JSInternalPromise* loadModule(JSGlobalObject*, const Identifier& moduleKey, JSValue parameters, JSValue scriptFetcher);
JS_EXPORT_PRIVATE JSInternalPromise* loadModule(JSGlobalObject*, const SourceCode&, JSValue scriptFetcher);

// Link and evaluate the already linked module. This function is called in a sync manner.
JS_EXPORT_PRIVATE JSValue linkAndEvaluateModule(JSGlobalObject*, const Identifier& moduleKey, JSValue scriptFetcher);

JS_EXPORT_PRIVATE JSInternalPromise* importModule(JSGlobalObject*, const Identifier& moduleName, JSValue referrer, JSValue parameters, JSValue scriptFetcher);

JS_EXPORT_PRIVATE UncheckedKeyHashMap<RefPtr<UniquedStringImpl>, String> retrieveImportAttributesFromDynamicImportOptions(JSGlobalObject*, JSValue, const Vector<RefPtr<UniquedStringImpl>>& supportedAssertions);
JS_EXPORT_PRIVATE std::optional<ScriptFetchParameters::Type> retrieveTypeImportAttribute(JSGlobalObject*, const UncheckedKeyHashMap<RefPtr<UniquedStringImpl>, String>&);

} // namespace JSC

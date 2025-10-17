/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 24, 2024.
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

#include "ErrorInstance.h"
#include "ErrorType.h"
#include "Exception.h"
#include "InternalFunction.h"
#include "JSObject.h"
#include "LineColumn.h"
#include "ThrowScope.h"
#include <stdint.h>


namespace JSC {

class CallFrame;
class VM;
class JSGlobalObject;
class JSObject;
class SourceCode;
class Structure;

JSObject* createError(JSGlobalObject*, const String&, ErrorInstance::SourceAppender);
JSObject* createEvalError(JSGlobalObject*, const String&, ErrorInstance::SourceAppender);
JSObject* createRangeError(JSGlobalObject*, const String&, ErrorInstance::SourceAppender);
JSObject* createReferenceError(JSGlobalObject*, const String&, ErrorInstance::SourceAppender);
JSObject* createSyntaxError(JSGlobalObject*, const String&, ErrorInstance::SourceAppender);
JSObject* createTypeError(JSGlobalObject*, const String&, ErrorInstance::SourceAppender, RuntimeType);
JSObject* createNotEnoughArgumentsError(JSGlobalObject*, ErrorInstance::SourceAppender);
JSObject* createURIError(JSGlobalObject*, const String&, ErrorInstance::SourceAppender);

JS_EXPORT_PRIVATE JSObject* createError(JSGlobalObject*, const String&);
JS_EXPORT_PRIVATE JSObject* createEvalError(JSGlobalObject*, const String&);
JS_EXPORT_PRIVATE JSObject* createRangeError(JSGlobalObject*, const String&);
JS_EXPORT_PRIVATE JSObject* createReferenceError(JSGlobalObject*, const String&);
JS_EXPORT_PRIVATE JSObject* createSyntaxError(JSGlobalObject*, const String&);
JS_EXPORT_PRIVATE JSObject* createSyntaxError(JSGlobalObject*);
JS_EXPORT_PRIVATE JSObject* createTypeError(JSGlobalObject*);
JS_EXPORT_PRIVATE JSObject* createTypeError(JSGlobalObject*, const String&);
JS_EXPORT_PRIVATE JSObject* createNotEnoughArgumentsError(JSGlobalObject*);
JS_EXPORT_PRIVATE JSObject* createURIError(JSGlobalObject*, const String&);
JS_EXPORT_PRIVATE JSObject* createOutOfMemoryError(JSGlobalObject*);
JS_EXPORT_PRIVATE JSObject* createOutOfMemoryError(JSGlobalObject*, const String&);

JS_EXPORT_PRIVATE JSObject* createError(JSGlobalObject*, ErrorType, const String&);
JS_EXPORT_PRIVATE JSObject* createError(JSGlobalObject*, ErrorTypeWithExtension, const String&);

std::unique_ptr<Vector<StackFrame>> getStackTrace(VM&, JSObject*, bool useCurrentFrame, JSCell* ownerOfCallLinkInfo = nullptr, CallLinkInfo* = nullptr);
std::tuple<CodeBlock*, BytecodeIndex> getBytecodeIndex(VM&, CallFrame*);
bool getLineColumnAndSource(VM&, Vector<StackFrame>* stackTrace, LineColumn&, String& sourceURL);
bool addErrorInfo(VM&, Vector<StackFrame>*, JSObject*);
JS_EXPORT_PRIVATE void addErrorInfo(JSGlobalObject*, JSObject*, bool);
JSObject* addErrorInfo(VM&, JSObject* error, int line, const SourceCode&);

// https://github.com/tc39/proposal-shadowrealm/pull/382
//
// When an crosses the ShadowRealm barrier, it is converted to a TypeError,
// without invoking any observable operations. It attempts to maintain the
// error message of the original error, if possible, and may include additional
// information.
JSObject* createTypeErrorCopy(JSGlobalObject*, JSValue error);

// Methods to throw Errors.

// Convenience wrappers, create an throw an exception with a default message.
JS_EXPORT_PRIVATE Exception* throwConstructorCannotBeCalledAsFunctionTypeError(JSGlobalObject*, ThrowScope&, ASCIILiteral constructorName);
JS_EXPORT_PRIVATE Exception* throwTypeError(JSGlobalObject*, ThrowScope&);
JS_EXPORT_PRIVATE Exception* throwTypeError(JSGlobalObject*, ThrowScope&, ASCIILiteral errorMessage);
JS_EXPORT_PRIVATE Exception* throwTypeError(JSGlobalObject*, ThrowScope&, const String& errorMessage);
JS_EXPORT_PRIVATE Exception* throwSyntaxError(JSGlobalObject*, ThrowScope&);
JS_EXPORT_PRIVATE Exception* throwSyntaxError(JSGlobalObject*, ThrowScope&, const String& errorMessage);
inline Exception* throwRangeError(JSGlobalObject* globalObject, ThrowScope& scope, const String& errorMessage) { return throwException(globalObject, scope, createRangeError(globalObject, errorMessage)); }

JS_EXPORT_PRIVATE String makeDOMAttributeGetterTypeErrorMessage(const char* interfaceName, const String& attributeName);
JS_EXPORT_PRIVATE String makeDOMAttributeSetterTypeErrorMessage(const char* interfaceName, const String& attributeName);

JS_EXPORT_PRIVATE JSValue throwDOMAttributeGetterTypeError(JSGlobalObject*, ThrowScope&, const ClassInfo*, PropertyName);
JS_EXPORT_PRIVATE JSValue throwDOMAttributeSetterTypeError(JSGlobalObject*, ThrowScope&, const ClassInfo*, PropertyName);

// Convenience wrappers, wrap result as an EncodedJSValue.
inline void throwVMError(JSGlobalObject* globalObject, ThrowScope& scope, Exception* exception) { throwException(globalObject, scope, exception); }
inline EncodedJSValue throwVMError(JSGlobalObject* globalObject, ThrowScope& scope, JSValue error) { return JSValue::encode(throwException(globalObject, scope, error)); }
inline EncodedJSValue throwVMError(JSGlobalObject* globalObject, ThrowScope& scope, const String& errorMessage) { return JSValue::encode(throwException(globalObject, scope, createError(globalObject, errorMessage))); }
inline EncodedJSValue throwVMTypeError(JSGlobalObject* globalObject, ThrowScope& scope) { return JSValue::encode(throwTypeError(globalObject, scope)); }
inline EncodedJSValue throwVMTypeError(JSGlobalObject* globalObject, ThrowScope& scope, ASCIILiteral errorMessage) { return JSValue::encode(throwTypeError(globalObject, scope, errorMessage)); }
inline EncodedJSValue throwVMTypeError(JSGlobalObject* globalObject, ThrowScope& scope, const String& errorMessage) { return JSValue::encode(throwTypeError(globalObject, scope, errorMessage)); }
inline EncodedJSValue throwVMRangeError(JSGlobalObject* globalObject, ThrowScope& scope, const String& errorMessage) { return JSValue::encode(throwRangeError(globalObject, scope, errorMessage)); }
inline EncodedJSValue throwVMDOMAttributeGetterTypeError(JSGlobalObject* globalObject, ThrowScope& scope, const ClassInfo* classInfo, PropertyName propertyName) { return JSValue::encode(throwDOMAttributeGetterTypeError(globalObject, scope, classInfo, propertyName)); }
inline EncodedJSValue throwVMDOMAttributeSetterTypeError(JSGlobalObject* globalObject, ThrowScope& scope, const ClassInfo* classInfo, PropertyName propertyName) { return JSValue::encode(throwDOMAttributeSetterTypeError(globalObject, scope, classInfo, propertyName)); }

} // namespace JSC

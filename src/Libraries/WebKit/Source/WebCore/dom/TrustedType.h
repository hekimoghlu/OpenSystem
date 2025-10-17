/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 3, 2024.
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

#include "ExceptionOr.h"
#include "TrustedHTML.h"
#include "TrustedScript.h"
#include "TrustedScriptURL.h"

namespace JSC {

class ArgList;

enum class CompilationType;

} // namespace JSC

namespace WebCore {

class Exception;
class ScriptExecutionContext;
class QualifiedName;

enum class TrustedType : int8_t {
    TrustedHTML,
    TrustedScript,
    TrustedScriptURL,
};

struct AttributeTypeAndSink {
    String attributeType;
    String sink;
};

ASCIILiteral trustedTypeToString(TrustedType);
TrustedType stringToTrustedType(String);
ASCIILiteral trustedTypeToCallbackName(TrustedType);

WEBCORE_EXPORT std::variant<std::monostate, Exception, Ref<TrustedHTML>, Ref<TrustedScript>, Ref<TrustedScriptURL>> processValueWithDefaultPolicy(ScriptExecutionContext&, TrustedType, const String& input, const String& sink);

WEBCORE_EXPORT ExceptionOr<String> trustedTypeCompliantString(TrustedType, ScriptExecutionContext&, const String& input, const String& sink);

WEBCORE_EXPORT ExceptionOr<String> requireTrustedTypesForPreNavigationCheckPasses(ScriptExecutionContext&, const String& urlString);

ExceptionOr<String> trustedTypeCompliantString(ScriptExecutionContext&, std::variant<RefPtr<TrustedHTML>, String>&&, const String& sink);

ExceptionOr<String> trustedTypeCompliantString(ScriptExecutionContext&, std::variant<RefPtr<TrustedScript>, String>&&, const String& sink);

ExceptionOr<String> trustedTypeCompliantString(ScriptExecutionContext&, std::variant<RefPtr<TrustedScriptURL>, String>&&, const String& sink);

WEBCORE_EXPORT AttributeTypeAndSink trustedTypeForAttribute(const String& elementName, const String& attributeName, const String& elementNamespace, const String& attributeNamespace);

ExceptionOr<bool> canCompile(ScriptExecutionContext&, JSC::CompilationType, String codeString, const JSC::ArgList& args);

bool isEventHandlerAttribute(const QualifiedName& attributeName);

} // namespace WebCore

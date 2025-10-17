/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 21, 2022.
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
#include "TrustedType.h"

#include "ContentSecurityPolicy.h"
#include "Document.h"
#include "EventNames.h"
#include "HTMLElement.h"
#include "JSDOMExceptionHandling.h"
#include "JSTrustedScript.h"
#include "LocalDOMWindow.h"
#include "SVGNames.h"
#include "TrustedTypePolicy.h"
#include "TrustedTypePolicyFactory.h"
#include "WindowOrWorkerGlobalScopeTrustedTypes.h"
#include "WorkerGlobalScope.h"
#include "XLinkNames.h"
#include <JavaScriptCore/ArgList.h>
#include <JavaScriptCore/HeapInlines.h>
#include <JavaScriptCore/JSCInlines.h>
#include <JavaScriptCore/JSCJSValueInlines.h>
#include <JavaScriptCore/JSCast.h>
#include <pal/text/TextEncoding.h>
#include <wtf/text/MakeString.h>

namespace WebCore {
using namespace JSC;

struct TrustedTypeVisitor {
    String operator()(std::monostate)
    {
        return nullString();
    }
    String operator()(Exception)
    {
        return nullString();
    }
    String operator()(const Ref<TrustedHTML>& value)
    {
        return value->toString();
    }
    String operator()(const Ref<TrustedScript>& value)
    {
        return value->toString();
    }
    String operator()(const Ref<TrustedScriptURL>& value)
    {
        return value->toString();
    }
};

ASCIILiteral trustedTypeToString(TrustedType trustedType)
{
    switch (trustedType) {
    case TrustedType::TrustedHTML:
        return "TrustedHTML"_s;
    case TrustedType::TrustedScript:
        return "TrustedScript"_s;
    case TrustedType::TrustedScriptURL:
        return "TrustedScriptURL"_s;
    }

    ASSERT_NOT_REACHED();
    return { };
}

TrustedType stringToTrustedType(String str)
{
    if (str == "TrustedHTML"_s)
        return TrustedType::TrustedHTML;
    if (str == "TrustedScript"_s)
        return TrustedType::TrustedScript;
    if (str == "TrustedScriptURL"_s)
        return TrustedType::TrustedScriptURL;

    ASSERT_NOT_REACHED();
    return { };
}

ASCIILiteral trustedTypeToCallbackName(TrustedType trustedType)
{
    switch (trustedType) {
    case TrustedType::TrustedHTML:
        return "createHTML"_s;
    case TrustedType::TrustedScript:
        return "createScript"_s;
    case TrustedType::TrustedScriptURL:
        return "createScriptURL"_s;
    }

    ASSERT_NOT_REACHED();
    return { };
}

// https://w3c.github.io/trusted-types/dist/spec/#process-value-with-a-default-policy-algorithm
std::variant<std::monostate, Exception, Ref<TrustedHTML>, Ref<TrustedScript>, Ref<TrustedScriptURL>> processValueWithDefaultPolicy(ScriptExecutionContext& scriptExecutionContext, TrustedType expectedType, const String& input, const String& sink)
{
    RefPtr<TrustedTypePolicy> protectedPolicy;
    if (RefPtr document = dynamicDowncast<Document>(scriptExecutionContext)) {
        if (RefPtr window = document->domWindow())
            protectedPolicy = WindowOrWorkerGlobalScopeTrustedTypes::trustedTypes(*window)->defaultPolicy();
    } else if (RefPtr workerGlobalScope = dynamicDowncast<WorkerGlobalScope>(scriptExecutionContext))
        protectedPolicy = WindowOrWorkerGlobalScopeTrustedTypes::trustedTypes(*workerGlobalScope)->defaultPolicy();

    if (!protectedPolicy)
        return std::monostate();

    VM& vm = scriptExecutionContext.vm();

    auto jsExpectedType = JSC::jsString(vm, String(trustedTypeToString(expectedType)));
    auto jsSink = JSC::jsString(vm, sink);
    FixedVector<JSC::Strong<JSC::Unknown>> arguments({ { vm, jsExpectedType }, { vm, jsSink } });
    auto policyValueHolder = protectedPolicy->getPolicyValue(expectedType, input, WTFMove(arguments), IfMissing::ReturnNull);
    if (policyValueHolder.hasException())
        return { policyValueHolder.releaseException() };

    auto policyValue = policyValueHolder.releaseReturnValue();
    if (policyValue.isNull())
        return std::monostate();

    switch (expectedType) {
    case TrustedType::TrustedHTML:
        return { TrustedHTML::create(policyValue) };
    case TrustedType::TrustedScript:
        return { TrustedScript::create(policyValue) };
    case TrustedType::TrustedScriptURL:
        return { TrustedScriptURL::create(policyValue) };
    }

    ASSERT_NOT_REACHED();
    return std::monostate();
}

// https://w3c.github.io/trusted-types/dist/spec/#get-trusted-type-compliant-string-algorithm
ExceptionOr<String> trustedTypeCompliantString(TrustedType expectedType, ScriptExecutionContext& scriptExecutionContext, const String& input, const String& sink)
{
    String stringValue(input);

    if (!scriptExecutionContext.settingsValues().trustedTypesEnabled)
        return stringValue;

    CheckedPtr contentSecurityPolicy = scriptExecutionContext.checkedContentSecurityPolicy();

    auto requireTrustedTypes = contentSecurityPolicy && contentSecurityPolicy->requireTrustedTypesForSinkGroup("script"_s);

    if (!requireTrustedTypes)
        return stringValue;

    auto convertedInput = processValueWithDefaultPolicy(scriptExecutionContext, expectedType, stringValue, sink);
    if (std::holds_alternative<Exception>(convertedInput))
        return WTFMove(std::get<Exception>(convertedInput));

    if (!std::holds_alternative<std::monostate>(convertedInput)) {
        stringValue = std::visit(TrustedTypeVisitor { }, convertedInput);
        if (stringValue.isNull())
            convertedInput = std::monostate();
    }

    if (std::holds_alternative<std::monostate>(convertedInput)) {
        auto allowMissingTrustedTypes = contentSecurityPolicy->allowMissingTrustedTypesForSinkGroup(trustedTypeToString(expectedType), sink, "script"_s, stringValue);

        if (!allowMissingTrustedTypes)
            return Exception { ExceptionCode::TypeError, makeString("This assignment requires a "_s, trustedTypeToString(expectedType)) };
    }

    return stringValue;
}

ExceptionOr<String> trustedTypeCompliantString(ScriptExecutionContext& scriptExecutionContext, std::variant<RefPtr<TrustedHTML>, String>&& input, const String& sink)
{
    return WTF::switchOn(
        WTFMove(input),
        [&scriptExecutionContext, &sink](const String& string) -> ExceptionOr<String> {
            return trustedTypeCompliantString(TrustedType::TrustedHTML, scriptExecutionContext, string, sink);
        },
        [](const RefPtr<TrustedHTML>& html) -> ExceptionOr<String> {
            return html->toString();
        }
    );
}

ExceptionOr<String> trustedTypeCompliantString(ScriptExecutionContext& scriptExecutionContext, std::variant<RefPtr<TrustedScript>, String>&& input, const String& sink)
{
    return WTF::switchOn(
        WTFMove(input),
        [&scriptExecutionContext, &sink](const String& string) -> ExceptionOr<String> {
            return trustedTypeCompliantString(TrustedType::TrustedScript, scriptExecutionContext, string, sink);
        },
        [](const RefPtr<TrustedScript>& script) -> ExceptionOr<String> {
            return script->toString();
        }
    );
}

ExceptionOr<String> trustedTypeCompliantString(ScriptExecutionContext& scriptExecutionContext, std::variant<RefPtr<TrustedScriptURL>, String>&& input, const String& sink)
{
    return WTF::switchOn(
        WTFMove(input),
        [&scriptExecutionContext, &sink](const String& string) -> ExceptionOr<String> {
            return trustedTypeCompliantString(TrustedType::TrustedScriptURL, scriptExecutionContext, string, sink);
        },
        [](const RefPtr<TrustedScriptURL>& scriptURL) -> ExceptionOr<String> {
            return scriptURL->toString();
        }
    );
}

AttributeTypeAndSink trustedTypeForAttribute(const String& elementName, const String& attributeName, const String& elementNamespace, const String& attributeNamespace)
{
    AttributeTypeAndSink returnValues;
    auto localName = elementName.convertToASCIILowercase();

    AtomString elementNS = elementNamespace.isEmpty() ? HTMLNames::xhtmlNamespaceURI : AtomString(elementNamespace);
    AtomString attributeNS = attributeNamespace.isEmpty() ? nullAtom() : AtomString(attributeNamespace);

    QualifiedName element(nullAtom(), AtomString(localName), elementNS);
    QualifiedName attribute(nullAtom(), AtomString(attributeName), attributeNS);

    if (attributeNS.isNull() && !attributeName.isNull()) {
        if (isEventHandlerAttribute(attribute)) {
            returnValues.sink = makeString("Element "_s, attributeName);
            returnValues.attributeType = trustedTypeToString(TrustedType::TrustedScript);
            return returnValues;
        }
    }

    if (element.matches(HTMLNames::iframeTag) && attribute.matches(HTMLNames::srcdocAttr)) {
        returnValues.sink = "HTMLIFrameElement srcdoc"_s;
        returnValues.attributeType = trustedTypeToString(TrustedType::TrustedHTML);
    }
    if (element.matches(HTMLNames::scriptTag) && attribute.matches(HTMLNames::srcAttr)) {
        returnValues.sink = "HTMLScriptElement src"_s;
        returnValues.attributeType = trustedTypeToString(TrustedType::TrustedScriptURL);
    }
    if (element.matches(SVGNames::scriptTag) && (attribute.matches(SVGNames::hrefAttr) || attribute.matches(XLinkNames::hrefAttr))) {
        returnValues.sink = "SVGScriptElement href"_s;
        returnValues.attributeType = trustedTypeToString(TrustedType::TrustedScriptURL);
    }

    return returnValues;
}

// https://w3c.github.io/trusted-types/dist/spec/#require-trusted-types-for-pre-navigation-check
ExceptionOr<String> requireTrustedTypesForPreNavigationCheckPasses(ScriptExecutionContext& scriptExecutionContext, const String& urlString)
{
    auto sinkGroup = "script"_s;
    auto sink = "Location href"_s;
    auto expectedType = TrustedType::TrustedScript;

    auto contentSecurityPolicy = scriptExecutionContext.contentSecurityPolicy();

    auto requireTrustedTypes = contentSecurityPolicy && contentSecurityPolicy->requireTrustedTypesForSinkGroup(sinkGroup);

    if (!requireTrustedTypes || !scriptExecutionContext.settingsValues().trustedTypesEnabled)
        return String(urlString);

    const int javascriptSchemeLength = sizeof("javascript:") - 1;
    auto decodedURLString = PAL::decodeURLEscapeSequences(urlString);
    auto scriptSource = decodedURLString.substring(javascriptSchemeLength);

    auto convertedScriptSource = processValueWithDefaultPolicy(scriptExecutionContext, expectedType, scriptSource, sink);
    if (std::holds_alternative<Exception>(convertedScriptSource))
        return WTFMove(std::get<Exception>(convertedScriptSource));

    if (std::holds_alternative<std::monostate>(convertedScriptSource)) {
        auto allowMissingTrustedTypes = contentSecurityPolicy->allowMissingTrustedTypesForSinkGroup(trustedTypeToString(expectedType), sink, sinkGroup, scriptSource);

        if (!allowMissingTrustedTypes)
            return Exception { ExceptionCode::TypeError, makeString("This assignment requires a "_s, trustedTypeToString(expectedType)) };

        return String(urlString);
    }

    auto stringifiedConvertedScriptSource = std::visit(TrustedTypeVisitor { }, convertedScriptSource);

    auto newURL = URL(makeString("javascript:"_s, stringifiedConvertedScriptSource));
    return String(newURL.isValid()
        ? newURL.string()
        : nullString());
}

ExceptionOr<bool> canCompile(ScriptExecutionContext& scriptExecutionContext, JSC::CompilationType compilationType, String codeString, const JSC::ArgList& args)
{
    if (compilationType == CompilationType::Function) {
        VM& vm = scriptExecutionContext.vm();
        auto scope = DECLARE_THROW_SCOPE(vm);

        bool isTrusted = true;

        for (size_t i = 0; i < args.size(); i++) {
            auto arg = args.at(i);
            if (!arg.isObject()) {
                isTrusted = false;
                break;
            }
            if (auto trustedScript = JSTrustedScript::toWrapped(vm, arg)) {
                if (!trustedScript) {
                    isTrusted = false;
                    break;
                }
                auto argString = arg.toWTFString(scriptExecutionContext.globalObject());
                RETURN_IF_EXCEPTION(scope, Exception { ExceptionCode::ExistingExceptionError });
                if (trustedScript->toString() != argString) {
                    isTrusted = false;
                    break;
                }
            } else {
                isTrusted = false;
                break;
            }
        }

        if (isTrusted)
            return true;
    }

    auto sink = compilationType == CompilationType::Function ? "Function"_s : "eval"_s;

    auto stringValueHolder = trustedTypeCompliantString(TrustedType::TrustedScript, scriptExecutionContext, codeString, sink);
    if (stringValueHolder.hasException())
        return stringValueHolder.releaseException();

    return codeString == stringValueHolder.releaseReturnValue();
}

bool isEventHandlerAttribute(const QualifiedName& attributeName)
{
    ASSERT(!attributeName.localName().isNull());

    // Event handler attributes have no namespace.
    if (!attributeName.namespaceURI().isNull())
        return false;

    // Fast early return for names that don't start with "on".
    AtomStringImpl& localName = *attributeName.localName().impl();
    if (localName.length() < 3 || localName[0] != 'o' || localName[1] != 'n')
        return false;
    static const NeverDestroyed<WTF::HashSet<AtomString>> eventHandlerNames([] {
        return eventNames().allEventHandlerNames();
    }());

    return eventHandlerNames->contains(&localName);
}

} // namespace WebCore

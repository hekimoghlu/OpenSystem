/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 18, 2024.
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
#include "URLPatternComponent.h"

#include "ScriptExecutionContext.h"
#include "URLPatternCanonical.h"
#include "URLPatternParser.h"
#include "URLPatternResult.h"
#include <JavaScriptCore/JSCJSValue.h>
#include <JavaScriptCore/JSString.h>
#include <JavaScriptCore/RegExpObject.h>

namespace WebCore {
using namespace JSC;
namespace URLPatternUtilities {

URLPatternComponent::URLPatternComponent(String&& patternString, JSC::Strong<JSC::RegExp>&& regex, Vector<String>&& groupNameList, bool hasRegexpGroupsFromPartsList)
    : m_patternString(WTFMove(patternString))
    , m_regularExpression(WTFMove(regex))
    , m_groupNameList(WTFMove(groupNameList))
    , m_hasRegexGroupsFromPartList(hasRegexpGroupsFromPartsList)
{
}

// https://urlpattern.spec.whatwg.org/#compile-a-component
ExceptionOr<URLPatternComponent> URLPatternComponent::compile(Ref<JSC::VM> vm, StringView input, EncodingCallbackType type, const URLPatternStringOptions& options)
{
    auto maybePartList = URLPatternParser::parse(input, options, type);
    if (maybePartList.hasException())
        return maybePartList.releaseException();
    Vector<Part> partList = maybePartList.releaseReturnValue();

    auto [regularExpressionString, nameList] = generateRegexAndNameList(partList, options);

    OptionSet<JSC::Yarr::Flags> flags = { JSC::Yarr::Flags::UnicodeSets };
    if (options.ignoreCase)
        flags.add(JSC::Yarr::Flags::IgnoreCase);

    JSC::RegExp* regularExpression = JSC::RegExp::create(vm, regularExpressionString, flags);
    if (!regularExpression->isValid())
        return Exception { ExceptionCode::TypeError, "Unable to create RegExp object regular expression from provided URLPattern string."_s };

    String patternString = generatePatternString(partList, options);

    bool hasRegexGroups = partList.containsIf([](auto& part) {
        return part.type == PartType::Regexp;
    });

    return URLPatternComponent { WTFMove(patternString), JSC::Strong<JSC::RegExp> { vm, regularExpression }, WTFMove(nameList), hasRegexGroups };
}

// https://urlpattern.spec.whatwg.org/#protocol-component-matches-a-special-scheme
bool URLPatternComponent::matchSpecialSchemeProtocol(ScriptExecutionContext& context) const
{
    Ref vm = context.vm();
    JSC::JSLockHolder lock(vm);

    static constexpr std::array specialSchemeList { "ftp"_s, "file"_s, "http"_s, "https"_s, "ws"_s, "wss"_s };
    auto protocolRegex = JSC::RegExpObject::create(vm, context.globalObject()->regExpStructure(), m_regularExpression.get(), true);

    auto isSchemeMatch = std::find_if(specialSchemeList.begin(), specialSchemeList.end(), [context = Ref { context }, &vm, &protocolRegex](const String& scheme) {
        auto maybeMatch = protocolRegex->exec(context->globalObject(), JSC::jsString(vm, scheme));
        return !maybeMatch.isNull();
    });

    return isSchemeMatch != specialSchemeList.end();
}

JSC::JSValue URLPatternComponent::componentExec(ScriptExecutionContext& context, StringView comparedString) const
{
    Ref vm = context.vm();
    JSC::JSLockHolder lock(vm);

    auto regex = JSC::RegExpObject::create(vm, context.globalObject()->regExpStructure(), m_regularExpression.get(), true);
    return regex->exec(context.globalObject(), JSC::jsString(vm, comparedString));
}

// https://urlpattern.spec.whatwg.org/#create-a-component-match-result
URLPatternComponentResult URLPatternComponent::createComponentMatchResult(ScriptExecutionContext& context, String&& input, const JSC::JSValue& execResult) const
{
    URLPatternComponentResult::GroupsRecord groups;

    auto globalObject = context.globalObject();
    Ref vm = globalObject->vm();

    auto length = execResult.get(globalObject, vm->propertyNames->length).toIntegerOrInfinity(globalObject);
    ASSERT(length >= 0 && std::isfinite(length));

    for (unsigned index = 1; index < length; ++index) {
        auto match = execResult.get(globalObject, index);

        std::variant<std::monostate, String> value;
        if (!match.isNull() && !match.isUndefined())
            value = match.toWTFString(globalObject);

        groups.append(URLPatternComponentResult::NameMatchPair { m_groupNameList[index - 1], WTFMove(value) });
    }

    return URLPatternComponentResult { !input.isEmpty() ? WTFMove(input) : emptyString(), WTFMove(groups) };
}

}
}

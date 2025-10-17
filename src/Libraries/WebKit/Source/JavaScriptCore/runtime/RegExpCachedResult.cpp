/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 15, 2022.
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
#include "RegExpCachedResult.h"

#include "RegExpCache.h"
#include "RegExpMatchesArray.h"

namespace JSC {

template<typename Visitor>
void RegExpCachedResult::visitAggregateImpl(Visitor& visitor)
{
    visitor.append(m_lastInput);
    visitor.append(m_lastRegExp);
    if (m_reified) {
        visitor.append(m_reifiedInput);
        visitor.append(m_reifiedResult);
        visitor.append(m_reifiedLeftContext);
        visitor.append(m_reifiedRightContext);
    }
}

DEFINE_VISIT_AGGREGATE(RegExpCachedResult);

JSArray* RegExpCachedResult::lastResult(JSGlobalObject* globalObject, JSObject* owner)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    if (!m_reified) {
        m_reifiedInput.set(vm, owner, m_lastInput.get());
        if (!m_lastRegExp)
            m_lastRegExp.set(vm, owner, vm.regExpCache()->ensureEmptyRegExp(vm));

        JSArray* result = nullptr;
        if (m_result) {
            auto* string = m_lastInput.get();
            auto input = string->view(globalObject);
            RETURN_IF_EXCEPTION(scope, nullptr);

            if (m_oneCharacterMatch) {
                ASSERT(m_lastRegExp->hasValidAtom());
                const String& pattern = m_lastRegExp->atom();
                ASSERT(!pattern.isEmpty());
                ASSERT(pattern.length() == 1);
                // Reify precise m_result.
                size_t found = input->reverseFind(pattern.characterAt(0));
                if (found != notFound) {
                    m_result.start = found;
                    m_result.end = found + 1;
                }
                m_oneCharacterMatch = false;
            }

            MatchResult ignoreMatched;
            result = createRegExpMatchesArray(vm, globalObject, string, input, m_lastRegExp.get(), m_result.start, ignoreMatched);
            RETURN_IF_EXCEPTION(scope, nullptr);
        } else {
            result = createEmptyRegExpMatchesArray(globalObject, m_lastInput.get(), m_lastRegExp.get());
            RETURN_IF_EXCEPTION(scope, nullptr);
        }

        m_reifiedResult.setWithoutWriteBarrier(result);
        m_reifiedLeftContext.clear();
        m_reifiedRightContext.clear();
        m_reified = true;
        vm.writeBarrier(owner);
    }
    return m_reifiedResult.get();
}

JSString* RegExpCachedResult::leftContext(JSGlobalObject* globalObject, JSObject* owner)
{
    // Make sure we're reified.
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    lastResult(globalObject, owner);
    RETURN_IF_EXCEPTION(scope, nullptr);

    if (!m_reifiedLeftContext) {
        JSString* leftContext = jsSubstring(globalObject, m_reifiedInput.get(), 0, m_result.start);
        RETURN_IF_EXCEPTION(scope, nullptr);
        m_reifiedLeftContext.set(vm, owner, leftContext);
    }
    return m_reifiedLeftContext.get();
}

JSString* RegExpCachedResult::rightContext(JSGlobalObject* globalObject, JSObject* owner)
{
    // Make sure we're reified.
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    lastResult(globalObject, owner);
    RETURN_IF_EXCEPTION(scope, nullptr);

    if (!m_reifiedRightContext) {
        unsigned length = m_reifiedInput->length();
        JSString* rightContext = jsSubstring(globalObject, m_reifiedInput.get(), m_result.end, length - m_result.end);
        RETURN_IF_EXCEPTION(scope, nullptr);
        m_reifiedRightContext.set(vm, owner, rightContext);
    }
    return m_reifiedRightContext.get();
}

void RegExpCachedResult::setInput(JSGlobalObject* globalObject, JSObject* owner, JSString* input)
{
    // Make sure we're reified, otherwise m_reifiedInput will be ignored.
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    lastResult(globalObject, owner);
    RETURN_IF_EXCEPTION(scope, void());
    leftContext(globalObject, owner);
    RETURN_IF_EXCEPTION(scope, void());
    rightContext(globalObject, owner);
    RETURN_IF_EXCEPTION(scope, void());
    ASSERT(m_reified);
    m_reifiedInput.set(vm, owner, input);
}

} // namespace JSC

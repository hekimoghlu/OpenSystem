/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 16, 2023.
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
#include "RegExpSubstringGlobalAtomCache.h"

#include "JSGlobalObjectInlines.h"
#include "RegExpObjectInlines.h"

namespace JSC {

template<typename Visitor>
void RegExpSubstringGlobalAtomCache::visitAggregateImpl(Visitor& visitor)
{
    visitor.append(m_lastSubstringBase);
    visitor.append(m_lastRegExp);
}

DEFINE_VISIT_AGGREGATE(RegExpSubstringGlobalAtomCache);

JSValue RegExpSubstringGlobalAtomCache::collectMatches(JSGlobalObject* globalObject, JSRopeString* substring, RegExp* regExp)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    const String& pattern = regExp->atom();
    ASSERT(!pattern.isEmpty());

    JSString* substringBase = substring->substringBase();
    unsigned substringOffset = substring->substringOffset();
    unsigned substringLength = substring->length();

    // Try to get the last cache if possible
    size_t numberOfMatches = 0;
    size_t startIndex = 0;
    MatchResult lastResult { 0, pattern.length() };
    ([&]() ALWAYS_INLINE_LAMBDA {
        if (regExp != m_lastRegExp.get())
            return;
        if (substringBase != m_lastSubstringBase.get())
            return;
        if (substringOffset != m_lastSubstringOffset)
            return;
        if (substringLength < m_lastSubstringLength)
            return;
        numberOfMatches = m_lastNumberOfMatches;
        startIndex = m_lastMatchEnd;
        lastResult = m_lastResult;
    })();

    // Keep the substring info above since the following may resolve the substring to a non-rope.
    auto input = substring->view(globalObject);
    RETURN_IF_EXCEPTION(scope, { });

    auto regExpMatch = [&]() ALWAYS_INLINE_LAMBDA {
        MatchResult result = globalObject->regExpGlobalData().performMatch(globalObject, regExp, substring, input, startIndex);
        if (UNLIKELY(scope.exception()))
            return;

        while (result) {
            lastResult = result;
            if (UNLIKELY(numberOfMatches > MAX_STORAGE_VECTOR_LENGTH)) {
                throwOutOfMemoryError(globalObject, scope);
                return;
            }

            numberOfMatches++;
            startIndex = result.end;
            if (result.empty())
                startIndex++;

            result = globalObject->regExpGlobalData().performMatch(globalObject, regExp, substring, input, startIndex);
            if (UNLIKELY(scope.exception()))
                return;
        }
    };

    bool oneCharacterMatch = false;
    if (pattern.is8Bit()) {
        if (input->is8Bit()) {
            if (pattern.length() == 1) {
                if (input->length() >= startIndex) {
                    oneCharacterMatch = true;
                    numberOfMatches += WTF::countMatchedCharacters(input->span8().subspan(startIndex), pattern.span8()[0]);
                    startIndex = input->length(); // Because the pattern atom is one character, it is ensured that we no longer find anything until this input string's end.
                }
            } else {
                regExpMatch();
                if (UNLIKELY(scope.exception()))
                    return { };
            }
        } else {
            if (pattern.length() == 1) {
                if (input->length() >= startIndex) {
                    oneCharacterMatch = true;
                    numberOfMatches += WTF::countMatchedCharacters(input->span16().subspan(startIndex), pattern.characterAt(0));
                    startIndex = input->length(); // Because the pattern atom is one character, it is ensured that we no longer find anything until this input string's end.
                }
            } else {
                regExpMatch();
                if (UNLIKELY(scope.exception()))
                    return { };
            }
        }
    } else {
        if (input->is8Bit()) {
            regExpMatch();
            if (UNLIKELY(scope.exception()))
                return { };
        } else {
            if (pattern.length() == 1) {
                if (input->length() >= startIndex) {
                    oneCharacterMatch = true;
                    numberOfMatches += WTF::countMatchedCharacters(input->span16().subspan(startIndex), pattern.characterAt(0));
                    startIndex = input->length(); // Because the pattern atom is one character, it is ensured that we no longer find anything until this input string's end.
                }
            } else {
                regExpMatch();
                if (UNLIKELY(scope.exception()))
                    return { };
            }
        }
    }

    if (UNLIKELY(numberOfMatches > MAX_STORAGE_VECTOR_LENGTH)) {
        throwOutOfMemoryError(globalObject, scope);
        return { };
    }

    if (!numberOfMatches)
        return jsNull();

    // Construct the array
    JSArray* array = createPatternFilledArray(globalObject, jsString(vm, pattern), numberOfMatches);
    RETURN_IF_EXCEPTION(scope, { });

    globalObject->regExpGlobalData().recordMatch(vm, globalObject, regExp, substring, lastResult, oneCharacterMatch);
    RETURN_IF_EXCEPTION(scope, { });

    // Cache
    {
        m_lastSubstringBase.set(vm, globalObject, substringBase);
        m_lastSubstringOffset = substringOffset;
        m_lastSubstringLength = substringLength;

        m_lastRegExp.set(vm, globalObject, regExp);
        m_lastNumberOfMatches = numberOfMatches;
        m_lastMatchEnd = startIndex;
        m_lastResult = lastResult;
    }
    return array;
}

} // namespace JSC

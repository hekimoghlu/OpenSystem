/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 5, 2024.
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

#include "ExceptionHelpers.h"
#include "JSCJSValue.h"
#include "JSGlobalObject.h"

namespace JSC {

class JSStringJoiner {
    WTF_FORBID_HEAP_ALLOCATION;
public:

    struct Entry {
        NO_UNIQUE_ADDRESS StringViewWithUnderlyingString m_view;
        NO_UNIQUE_ADDRESS uint16_t m_additional { 0 };
    };
    using Entries = Vector<Entry, 16>;

    JSStringJoiner(StringView separator);
    ~JSStringJoiner();

    void reserveCapacity(JSGlobalObject*, size_t);

    void append(JSGlobalObject*, JSValue);
    void appendNumber(VM&, int32_t);
    void appendNumber(VM&, double);
    bool appendWithoutSideEffects(JSGlobalObject*, JSValue);
    void appendEmptyString();

    JSValue join(JSGlobalObject*);

private:
    void append(JSString*, StringViewWithUnderlyingString&&);
    void append8Bit(const String&);
    unsigned joinedLength(JSGlobalObject*) const;
    JSValue joinSlow(JSGlobalObject*);

    StringView m_separator;
    Entries m_strings;
    CheckedUint32 m_accumulatedStringsLength;
    CheckedUint32 m_stringsCount;
    bool m_hasOverflowed { false };
    bool m_isAll8Bit { true };
    JSString* m_lastString { nullptr };
};

inline JSStringJoiner::JSStringJoiner(StringView separator)
    : m_separator(separator)
    , m_isAll8Bit(m_separator.is8Bit())
{
}

inline void JSStringJoiner::reserveCapacity(JSGlobalObject* globalObject, size_t count)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    if (UNLIKELY(!m_strings.tryReserveCapacity(count)))
        throwOutOfMemoryError(globalObject, scope);
}

inline JSValue JSStringJoiner::join(JSGlobalObject* globalObject)
{
    if (m_stringsCount == 1) {
        // If m_stringsCount is 1, then there's no chance of an overflow because m_strings
        // is a Vector<Entry, 16>, and has at least space for 16 entries.
        ASSERT(!m_hasOverflowed);
        if (m_lastString)
            return m_lastString;
        return jsString(globalObject->vm(), m_strings[0].m_view.toString());
    }
    return joinSlow(globalObject);
}

ALWAYS_INLINE void JSStringJoiner::append(JSString* jsString, StringViewWithUnderlyingString&& string)
{
    ++m_stringsCount;
    if (m_lastString == jsString) {
        auto& entry = m_strings.last();
        if (LIKELY(entry.m_additional < UINT16_MAX)) {
            ++entry.m_additional;
            m_accumulatedStringsLength += entry.m_view.view.length();
            return;
        }
    }
    m_accumulatedStringsLength += string.view.length();
    m_isAll8Bit = m_isAll8Bit && string.view.is8Bit();
    m_hasOverflowed |= !m_strings.tryAppend({ WTFMove(string), 0 });
    m_lastString = jsString;
}

ALWAYS_INLINE void JSStringJoiner::append8Bit(const String& string)
{
    ASSERT(string.is8Bit());
    ++m_stringsCount;
    m_accumulatedStringsLength += string.length();
    m_hasOverflowed |= !m_strings.tryAppend({ { string, string }, 0 });
    m_lastString = nullptr;
}

ALWAYS_INLINE void JSStringJoiner::appendEmptyString()
{
    ++m_stringsCount;
    m_hasOverflowed |= !m_strings.tryAppend({ { { }, { } }, 0 });
    m_lastString = nullptr;
}

ALWAYS_INLINE bool JSStringJoiner::appendWithoutSideEffects(JSGlobalObject* globalObject, JSValue value)
{
    // The following code differs from using the result of JSValue::toString in the following ways:
    // 1) It's inlined more than JSValue::toString is.
    // 2) It includes conversion to WTF::String in a way that avoids allocating copies of substrings.
    // 3) It doesn't create a JSString for numbers, true, or false.
    // 4) It turns undefined and null into the empty string instead of "undefined" and "null".
    // 5) It uses optimized code paths for all the cases known to be 8-bit and for the empty string.
    // If we might make an effectful calls, return false. Otherwise return true.

    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    if (value.isCell()) {
        // FIXME: Support JSBigInt in side-effect-free append.
        // https://bugs.webkit.org/show_bug.cgi?id=211173
        if (JSString* jsString = jsDynamicCast<JSString*>(value)) {
            auto view = jsString->view(globalObject);
            RETURN_IF_EXCEPTION(scope, false);
            // Since getting the view didn't OOM, we know that the underlying String exists and isn't
            // a rope. Thus, `tryGetValue` on the owner JSString will succeed. Since jsString could be
            // a substring we make sure to get the owner's String not jsString's.
            append(jsString, StringViewWithUnderlyingString(view, jsCast<const JSString*>(view.owner)->tryGetValue()));
            return true;
        }
        return false;
    }

    if (value.isInt32()) {
        appendNumber(globalObject->vm(), value.asInt32());
        return true;
    }
    if (value.isDouble()) {
        appendNumber(globalObject->vm(), value.asDouble());
        return true;
    }
    if (value.isTrue()) {
        append8Bit(globalObject->vm().propertyNames->trueKeyword.string());
        return true;
    }
    if (value.isFalse()) {
        append8Bit(globalObject->vm().propertyNames->falseKeyword.string());
        return true;
    }

#if USE(BIGINT32)
    if (value.isBigInt32()) {
        appendNumber(globalObject->vm(), value.bigInt32AsInt32());
        return true;
    }
#endif

    ASSERT(value.isUndefinedOrNull());
    appendEmptyString();
    return true;
}

ALWAYS_INLINE void JSStringJoiner::append(JSGlobalObject* globalObject, JSValue value)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    bool success = appendWithoutSideEffects(globalObject, value);
    RETURN_IF_EXCEPTION(scope, void());
    if (!success) {
        ASSERT(value.isCell());
        ASSERT(!value.isString());
        JSString* jsString = value.asCell()->toStringInline(globalObject);
        RETURN_IF_EXCEPTION(scope, void());
        auto view = jsString->view(globalObject);
        RETURN_IF_EXCEPTION(scope, void());
        RELEASE_AND_RETURN(scope, append(jsString, StringViewWithUnderlyingString(view, jsCast<const JSString*>(view.owner)->tryGetValue())));
    }
}

ALWAYS_INLINE void JSStringJoiner::appendNumber(VM& vm, int32_t value)
{
    append8Bit(vm.numericStrings.add(value));
}

ALWAYS_INLINE void JSStringJoiner::appendNumber(VM& vm, double value)
{
    if (canBeStrictInt32(value))
        appendNumber(vm, static_cast<int32_t>(value));
    else
        append8Bit(vm.numericStrings.add(value));
}

} // namespace JSC

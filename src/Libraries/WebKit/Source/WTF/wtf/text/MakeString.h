/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 4, 2023.
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

#include <wtf/CheckedArithmetic.h>
#include <wtf/StdLibExtras.h>
#include <wtf/text/AtomString.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/text/StringConcatenateNumbers.h>
#include <wtf/text/StringView.h>

namespace WTF {

template<typename StringTypeAdapter> constexpr bool makeStringSlowPathRequiredForAdapter = requires(const StringTypeAdapter& adapter) {
    { adapter.writeUsing(std::declval<StringBuilder&>) } -> std::same_as<void>;
};
template<typename... StringTypeAdapters> constexpr bool makeStringSlowPathRequired = (... || makeStringSlowPathRequiredForAdapter<StringTypeAdapters>);

template<typename... StringTypeAdapters>
RefPtr<StringImpl> tryMakeStringImplFromAdaptersInternal(unsigned length, bool areAllAdapters8Bit, StringTypeAdapters... adapters)
{
    ASSERT(length <= String::MaxLength);
    if (areAllAdapters8Bit) {
        std::span<LChar> buffer;
        RefPtr result = StringImpl::tryCreateUninitialized(length, buffer);
        if (!result)
            return nullptr;

        if (buffer.data())
            stringTypeAdapterAccumulator(buffer, adapters...);

        return result;
    }

    std::span<UChar> buffer;
    RefPtr result = StringImpl::tryCreateUninitialized(length, buffer);
    if (!result)
        return nullptr;

    if (buffer.data())
        stringTypeAdapterAccumulator(buffer, adapters...);

    return result;
}

template<typename... StringTypeAdapters>
String tryMakeStringFromAdapters(StringTypeAdapters&&... adapters)
{
    static_assert(String::MaxLength == std::numeric_limits<int32_t>::max());

    if constexpr (makeStringSlowPathRequired<StringTypeAdapters...>) {
        StringBuilder builder;
        builder.appendFromAdapters(std::forward<StringTypeAdapters>(adapters)...);
        return builder.toString();
    } else {
        auto sum = checkedSum<int32_t>(adapters.length()...);
        if (sum.hasOverflowed())
            return String();

        bool areAllAdapters8Bit = are8Bit(adapters...);
        return tryMakeStringImplFromAdaptersInternal(sum, areAllAdapters8Bit, adapters...);
    }
}

template<StringTypeAdaptable... StringTypes>
String tryMakeString(const StringTypes& ...strings)
{
    return tryMakeStringFromAdapters(StringTypeAdapter<StringTypes>(strings)...);
}

template<StringTypeAdaptable... StringTypes>
String makeString(StringTypes... strings)
{
    auto result = tryMakeString(strings...);
    if (!result)
        CRASH();
    return result;
}

template<typename... StringTypeAdapters>
AtomString tryMakeAtomStringFromAdapters(StringTypeAdapters ...adapters)
{
    static_assert(String::MaxLength == std::numeric_limits<int32_t>::max());

    if constexpr (makeStringSlowPathRequired<StringTypeAdapters...>) {
        StringBuilder builder;
        builder.appendFromAdapters(adapters...);
        return builder.toAtomString();
    } else {
        auto sum = checkedSum<int32_t>(adapters.length()...);
        if (sum.hasOverflowed())
            return AtomString();

        unsigned length = sum;
        ASSERT(length <= String::MaxLength);

        bool areAllAdapters8Bit = are8Bit(adapters...);
        constexpr size_t maxLengthToUseStackVariable = 64;
        if (length < maxLengthToUseStackVariable) {
            if (areAllAdapters8Bit) {
                std::array<LChar, maxLengthToUseStackVariable> buffer;
                stringTypeAdapterAccumulator(std::span<LChar> { buffer }, adapters...);
                return std::span<const LChar> { buffer }.first(length);
            }
            std::array<UChar, maxLengthToUseStackVariable> buffer;
            stringTypeAdapterAccumulator(std::span<UChar> { buffer }, adapters...);
            return std::span<const UChar> { buffer }.first(length);
        }
        return tryMakeStringImplFromAdaptersInternal(length, areAllAdapters8Bit, adapters...).get();
    }
}

template<StringTypeAdaptable... StringTypes>
AtomString tryMakeAtomString(StringTypes... strings)
{
    return tryMakeAtomStringFromAdapters(StringTypeAdapter<StringTypes>(strings)...);
}

template<StringTypeAdaptable... StringTypes>
AtomString makeAtomString(StringTypes... strings)
{
    auto result = tryMakeAtomString(strings...);
    if (result.isNull())
        CRASH();
    return result;
}

inline String WARN_UNUSED_RETURN makeStringByInserting(StringView originalString, StringView stringToInsert, unsigned position)
{
    return makeString(originalString.left(position), stringToInsert, originalString.substring(position));
}

// Helper functor useful in generic contexts where both makeString() and StringBuilder are being used.
struct SerializeUsingMakeString {
    using Result = String;
    template<typename... T> String operator()(T&&... args)
    {
        return makeString(std::forward<T>(args)...);
    }
};

} // namespace WTF

using WTF::makeAtomString;
using WTF::makeString;
using WTF::makeStringByInserting;
using WTF::tryMakeAtomString;
using WTF::tryMakeString;
using WTF::SerializeUsingMakeString;

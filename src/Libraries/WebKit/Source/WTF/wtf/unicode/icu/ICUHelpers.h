/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 16, 2025.
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

#include <tuple>
#include <unicode/utypes.h>
#include <wtf/Forward.h>
#include <wtf/FunctionTraits.h>

namespace WTF {

constexpr bool needsToGrowToProduceBuffer(UErrorCode);
constexpr bool needsToGrowToProduceCString(UErrorCode);

// Use this to call a function from ICU that has the following properties:
// - Takes a buffer pointer and capacity.
// - Returns the length of the buffer needed.
// - Takes a UErrorCode* as its last argument, returning the status, including U_BUFFER_OVERFLOW_ERROR.
// Pass the arguments, but pass a Vector in place of the buffer pointer and capacity, and don't pass a UErrorCode*.
// This will call the function, once or twice as needed, resizing the buffer as needed.
//
// Example:
//
//    Vector<UChar, 32> buffer;
//    auto status = callBufferProducingFunction(ucal_getDefaultTimeZone, buffer);
//
template<typename FunctionType, typename ...ArgumentTypes> UErrorCode callBufferProducingFunction(const FunctionType&, ArgumentTypes&&...);

// Implementations of the functions declared above.

constexpr bool needsToGrowToProduceBuffer(UErrorCode errorCode)
{
    return errorCode == U_BUFFER_OVERFLOW_ERROR;
}

constexpr bool needsToGrowToProduceCString(UErrorCode errorCode)
{
    return needsToGrowToProduceBuffer(errorCode) || errorCode == U_STRING_NOT_TERMINATED_WARNING;
}

namespace CallBufferProducingFunction {

template<typename CharacterType, size_t inlineCapacity, typename ...ArgumentTypes> auto& findVector(Vector<CharacterType, inlineCapacity>& buffer, ArgumentTypes&&...)
{
    return buffer;
}

template<typename FirstArgumentType, typename ...ArgumentTypes> auto& findVector(FirstArgumentType&&, ArgumentTypes&&... arguments)
{
    return findVector(std::forward<ArgumentTypes>(arguments)...);
}

constexpr std::tuple<> argumentTuple() { return { }; }

template<typename FirstArgumentType, typename ...OtherArgumentTypes> auto argumentTuple(FirstArgumentType&&, OtherArgumentTypes&&...);

template<typename CharacterType, size_t inlineCapacity, typename ...OtherArgumentTypes> auto argumentTuple(Vector<CharacterType, inlineCapacity>& buffer, OtherArgumentTypes&&... otherArguments)
{
    return tuple_cat(std::make_tuple(buffer.data(), buffer.size()), argumentTuple(std::forward<OtherArgumentTypes>(otherArguments)...));
}

template<typename FirstArgumentType, typename ...OtherArgumentTypes> auto argumentTuple(FirstArgumentType&& firstArgument, OtherArgumentTypes&&... otherArguments)
{
    // This technique of building a tuple and passing it twice does not work well for complex types, so assert this is a relatively simple one.
    static_assert(std::is_trivial_v<std::remove_reference_t<FirstArgumentType>>);
    return tuple_cat(std::make_tuple(firstArgument), argumentTuple(std::forward<OtherArgumentTypes>(otherArguments)...));
}

}

template<typename FunctionType, typename ...ArgumentTypes> UErrorCode callBufferProducingFunction(const FunctionType& function, ArgumentTypes&&... arguments)
{
    auto& buffer = CallBufferProducingFunction::findVector(std::forward<ArgumentTypes>(arguments)...);
    buffer.grow(buffer.capacity());
    auto status = U_ZERO_ERROR;
    auto resultLength = std::apply(function, CallBufferProducingFunction::argumentTuple(std::forward<ArgumentTypes>(arguments)..., &status));
    if (U_SUCCESS(status))
        buffer.shrink(resultLength);
    else if (needsToGrowToProduceBuffer(status)) {
        status = U_ZERO_ERROR;
        buffer.grow(resultLength);
        std::apply(function, CallBufferProducingFunction::argumentTuple(std::forward<ArgumentTypes>(arguments)..., &status));
        ASSERT(U_SUCCESS(status));
    }
    return status;
}

template<auto deleteFunction>
struct ICUDeleter {
    void operator()(typename FunctionTraits<decltype(deleteFunction)>::template ArgumentType<0> pointer)
    {
        if (pointer)
            deleteFunction(pointer);
    }
};

namespace ICU {

WTF_EXPORT_PRIVATE unsigned majorVersion();
WTF_EXPORT_PRIVATE unsigned minorVersion();

} // namespace ICU
} // namespace WTF

using WTF::callBufferProducingFunction;
using WTF::needsToGrowToProduceCString;
using WTF::needsToGrowToProduceBuffer;
using WTF::ICUDeleter;

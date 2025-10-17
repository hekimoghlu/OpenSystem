/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 13, 2022.
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
#include <type_traits>

namespace WTF {

template<typename T>
struct FunctionTraits;

#if USE(JSVALUE32_64)

template<typename T>
static constexpr unsigned slotsForCCallArgument()
{
    static_assert(!std::is_class<T>::value || sizeof(T) <= sizeof(void*), "This doesn't support complex structs.");
    static_assert(sizeof(T) == 8 || sizeof(T) <= 4);
    // This assumes that all integral values are passed on the stack.
    if (sizeof(T) == 8)
        return 2;

    return 1;
}

template<typename T>
static constexpr unsigned computeCCallSlots() { return slotsForCCallArgument<T>(); }

template<typename T, typename... Ts>
static constexpr std::enable_if_t<!!sizeof...(Ts), unsigned> computeCCallSlots() { return computeCCallSlots<Ts...>() + slotsForCCallArgument<T>(); }

#endif

template<typename Result, typename... Args>
struct FunctionTraits<Result(Args...)> {
    using ResultType = Result;

    static constexpr bool hasResult = !std::is_same<ResultType, void>::value;

    static constexpr std::size_t arity = sizeof...(Args);

    template <std::size_t n, typename = std::enable_if_t<(n < arity)>>
    using ArgumentType = typename std::tuple_element<n, std::tuple<Args...>>::type;
    using ArgumentTypes = std::tuple<Args...>;


#if USE(JSVALUE64)
    static constexpr unsigned cCallArity() { return arity; }
#else

    static constexpr unsigned cCallArity() { return computeCCallSlots<Args...>(); }
#endif // USE(JSVALUE64)

};

#if OS(WINDOWS)
template<typename Result, typename... Args>
struct FunctionTraits<Result SYSV_ABI(Args...)> : public FunctionTraits<Result(Args...)> {
};
#endif

template<typename Result, typename... Args>
struct FunctionTraits<Result(*)(Args...)> : public FunctionTraits<Result(Args...)> {
};

#if OS(WINDOWS)
template<typename Result, typename... Args>
struct FunctionTraits<Result SYSV_ABI (*)(Args...)> : public FunctionTraits<Result(Args...)> {
};
#endif

template<typename Result, typename... Args>
struct FunctionTraits<Result(Args...) noexcept> : public FunctionTraits<Result(Args...)> {
};

template<typename Result, typename... Args>
struct FunctionTraits<Result(*)(Args...) noexcept> : public FunctionTraits<Result(Args...)> {
};

} // namespace WTF

using WTF::FunctionTraits;

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 10, 2024.
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

#include "CSSParserToken.h"
#include "CSSParserTokenRange.h"
#include "CSSPrimitiveNumericTypes.h"
#include "CSSPropertyParserConsumer+MetaConsumerDefinitions.h"
#include "CSSPropertyParserOptions.h"
#include "StylePrimitiveNumericTypes.h"
#include <optional>
#include <type_traits>
#include <wtf/Brigand.h>
#include <wtf/StdLibExtras.h>

namespace WebCore {

enum CSSParserMode : uint8_t;
enum class ValueRange : uint8_t;

namespace CSSPropertyParserHelpers {

// MARK: - Meta Consumers

/// The result of a meta consume.
/// To be used with a list of `CSS` types (e.g. `ConsumeResult<CSS::Angle<Range>, CSS::Percentage<Range>, CSS::Keyword::None>`), which will yield a
/// result type of either a std::variant of those types (e.g.`std::variant<CSS::Angle<Range>, CSS::Percentage<Range>, CSS::Keyword::None>`) or the type
/// itself if only a single type was specified.
template<typename... Ts>
struct MetaConsumeResult {
    using TypeList = brigand::list<Ts...>;
    using type = VariantOrSingle<TypeList>;
};

template<CSSParserTokenType tokenType, typename Consumer, typename = void>
struct MetaConsumerDispatcher {
    static constexpr bool supported = false;
};

template<typename Consumer>
struct MetaConsumerDispatcher<FunctionToken, Consumer, typename std::void_t<typename Consumer::FunctionToken>> {
    static constexpr bool supported = true;
    template<typename... Args>
    static decltype(auto) consume(Args&&... args)
    {
        return Consumer::FunctionToken::consume(std::forward<Args>(args)...);
    }
};

template<typename Consumer>
struct MetaConsumerDispatcher<NumberToken, Consumer, typename std::void_t<typename Consumer::NumberToken>> {
    static constexpr bool supported = true;
    template<typename... Args>
    static decltype(auto) consume(Args&&... args)
    {
        return Consumer::NumberToken::consume(std::forward<Args>(args)...);
    }
};

template<typename Consumer>
struct MetaConsumerDispatcher<PercentageToken, Consumer, typename std::void_t<typename Consumer::PercentageToken>> {
    static constexpr bool supported = true;
    template<typename... Args>
    static decltype(auto) consume(Args&&... args)
    {
        return Consumer::PercentageToken::consume(std::forward<Args>(args)...);
    }
};

template<typename Consumer>
struct MetaConsumerDispatcher<DimensionToken, Consumer, typename std::void_t<typename Consumer::DimensionToken>> {
    static constexpr bool supported = true;
    template<typename... Args>
    static decltype(auto) consume(Args&&... args)
    {
        return Consumer::DimensionToken::consume(std::forward<Args>(args)...);
    }
};

template<typename Consumer>
struct MetaConsumerDispatcher<IdentToken, Consumer, typename std::void_t<typename Consumer::IdentToken>> {
    static constexpr bool supported = true;
    template<typename... Args>
    static decltype(auto) consume(Args&&... args)
    {
        return Consumer::IdentToken::consume(std::forward<Args>(args)...);
    }
};

// The `MetaConsumerUnroller` gives each type in the consumer list (`Ts...`)
// a chance to consume the token. It recursively peels off types from the
// type list, checks if the consumer supports this token type, and then calls
// to the MetaConsumerDispatcher to actually call right `consume` function.

// Empty case, used to indicate no more types remain to try.
template<typename... Ts>
struct MetaConsumerUnroller {
    template<CSSParserTokenType, typename ResultType>
    static std::nullopt_t consume(CSSParserTokenRange&, const CSSParserContext&, CSSCalcSymbolsAllowed, CSSPropertyParserOptions)
    {
        return std::nullopt;
    }

    template<CSSParserTokenType, typename ResultType, typename F>
    static std::nullopt_t consume(CSSParserTokenRange&, const CSSParserContext&, CSSCalcSymbolsAllowed, CSSPropertyParserOptions, NOESCAPE F&&)
    {
        return std::nullopt;
    }
};

// Actionable case, checks if the `Consumer` defined for type `T` supports the
// current token, trying to consume if it does, and in either case, falling
// back to recursively trying the same on the remainder of the type list `Ts...`.
template<typename T, typename... Ts>
struct MetaConsumerUnroller<T, Ts...> {
    template<CSSParserTokenType tokenType, typename ResultType>
    static std::optional<ResultType> consume(CSSParserTokenRange& range, const CSSParserContext& context, CSSCalcSymbolsAllowed symbolsAllowed, CSSPropertyParserOptions options)
    {
        using Consumer = MetaConsumerDispatcher<tokenType, ConsumerDefinition<T>>;
        if constexpr (Consumer::supported) {
            if (auto result = Consumer::consume(range, context, symbolsAllowed, options))
                return { T { *result } };
        }
        return MetaConsumerUnroller<Ts...>::template consume<tokenType, ResultType>(range, context, symbolsAllowed, options);
    }

    template<CSSParserTokenType tokenType, typename ResultType, typename F>
    static std::optional<ResultType> consume(CSSParserTokenRange& range, const CSSParserContext& context, CSSCalcSymbolsAllowed symbolsAllowed, CSSPropertyParserOptions options, NOESCAPE F&& functor)
    {
        using Consumer = MetaConsumerDispatcher<tokenType, ConsumerDefinition<T>>;
        if constexpr (Consumer::supported) {
            if (auto result = Consumer::consume(range, context, symbolsAllowed, options))
                return std::make_optional(functor(T { *result }));
        }
        return MetaConsumerUnroller<Ts...>::template consume<tokenType, ResultType>(range, context, symbolsAllowed, options, std::forward<F>(functor));
    }
};

// The `MetaConsumer` is the main driver of token consumption, dispatching
// to a `MetaConsumerUnroller` based on token type. Caller use this directly.
// An example use that attempts to consumer either a <number> or <percentage>
// looks like (argument list elided for brevity):
//
//    auto result = MetaConsumer<CSS::Percentage<R>, CSS::Number<R>>::consume(range, ...);
//
// If a caller wants to avoid the overhead of switching on the returned variant
// result, an alternative overload of `consume` is provided which takes an additional
// `functor` argument which gets called with the result:
//
//    auto result = MetaConsumer<CSS::Percentage<R>, CSS::Number<R>>::consume(range, ...,
//        [](CSS::Percentage<R> percentage) { ... },
//        [](CSS::Number<R> number) { ... }
//    );
template<typename T, typename... Ts>
struct MetaConsumer {
    using Unroller = MetaConsumerUnroller<T, Ts...>;

    template<typename... F>
    static decltype(auto) consume(CSSParserTokenRange& range, const CSSParserContext& context, CSSCalcSymbolsAllowed symbolsAllowed, CSSPropertyParserOptions options, F&&... f)
    {
        auto visitor = WTF::makeVisitor(std::forward<F>(f)...);
        using ResultType = decltype(visitor(std::declval<T>()));

        switch (range.peek().type()) {
        case FunctionToken:
            return Unroller::template consume<FunctionToken, ResultType>(range, context, WTFMove(symbolsAllowed), options, visitor);

        case NumberToken:
            return Unroller::template consume<NumberToken, ResultType>(range, context, WTFMove(symbolsAllowed), options, visitor);

        case PercentageToken:
            return Unroller::template consume<PercentageToken, ResultType>(range, context, WTFMove(symbolsAllowed), options, visitor);

        case DimensionToken:
            return Unroller::template consume<DimensionToken, ResultType>(range, context, WTFMove(symbolsAllowed), options, visitor);

        case IdentToken:
            return Unroller::template consume<IdentToken, ResultType>(range, context, WTFMove(symbolsAllowed), options, visitor);

        default:
            return std::optional<ResultType> { };
        }
    }

    static decltype(auto) consume(CSSParserTokenRange& range, const CSSParserContext& context, CSSCalcSymbolsAllowed symbolsAllowed, CSSPropertyParserOptions options)
    {
        using ResultType = typename MetaConsumeResult<T, Ts...>::type;

        return consume(range, context, WTFMove(symbolsAllowed), options,
            [](auto&& value) {
                return ResultType { WTFMove(value) };
            }
        );
    }
};

} // namespace CSSPropertyParserHelpers
} // namespace WebCore

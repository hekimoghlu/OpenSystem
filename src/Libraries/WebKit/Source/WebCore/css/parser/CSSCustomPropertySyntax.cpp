/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 18, 2022.
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
#include "CSSCustomPropertySyntax.h"

#include "CSSParserIdioms.h"
#include "CSSParserTokenRange.h"
#include "CSSTokenizer.h"
#include <wtf/SortedArrayMap.h>
#include <wtf/text/ParsingUtilities.h>

namespace WebCore {

template<typename CharacterType>
auto CSSCustomPropertySyntax::parseComponent(std::span<const CharacterType> span) -> std::optional<Component>
{
    StringParsingBuffer buffer { span };

    auto consumeMultiplier = [&] {
        if (skipExactly(buffer, '+'))
            return Multiplier::SpaceList;
        if (skipExactly(buffer, '#'))
            return Multiplier::CommaList;
        return Multiplier::Single;
    };

    if (skipExactly(buffer, '<')) {
        auto begin = buffer.span();
        skipUntil(buffer, '>');
        if (buffer.position() == begin.data())
            return { };

        auto dataTypeName = StringView(begin.first(buffer.position() - begin.data()));
        if (!skipExactly(buffer, '>'))
            return { };

        auto multiplier = consumeMultiplier();

        skipWhile<isCSSSpace>(buffer);
        if (!buffer.atEnd())
            return { };

        auto type = typeForTypeName(dataTypeName);

        // <transform-list> is a pre-multiplied data type.
        // https://drafts.css-houdini.org/css-properties-values-api/#multipliers
        if (type == Type::TransformList && multiplier != Multiplier::Single)
            type = Type::Unknown;

        return Component { type, multiplier };
    }

    auto begin = buffer.span();
    while (buffer.hasCharactersRemaining() && (*buffer != '+' && *buffer != '#'))
        ++buffer;

    auto ident = [&] {
        auto tokenizer = CSSTokenizer::tryCreate(StringView(begin.first(buffer.position() - begin.data())).toStringWithoutCopying());
        if (!tokenizer)
            return nullAtom();

        auto range = tokenizer->tokenRange();
        range.consumeWhitespace();
        if (range.peek().type() != IdentToken || !isValidCustomIdentifier(range.peek().id()))
            return nullAtom();

        auto value = range.consumeIncludingWhitespace().value();
        return range.atEnd() ? value.toAtomString() : nullAtom();
    }();

    if (ident.isNull())
        return { };

    auto multiplier = consumeMultiplier();

    return Component { Type::CustomIdent, multiplier, ident };
}

std::optional<CSSCustomPropertySyntax> CSSCustomPropertySyntax::parse(StringView syntax)
{
    // The format doesn't quite parse with CSSTokenizer.
    return readCharactersForParsing(syntax, [&](auto buffer) -> std::optional<CSSCustomPropertySyntax> {
        skipWhile<isCSSSpace>(buffer);

        if (skipExactly(buffer, '*')) {
            skipWhile<isCSSSpace>(buffer);
            if (buffer.hasCharactersRemaining())
                return { };

            return universal();
        }

        Definition definition;

        while (buffer.hasCharactersRemaining()) {
            auto begin = buffer.span();

            skipUntil(buffer, '|');

            auto component = parseComponent(begin.first(buffer.position() - begin.data()));
            if (!component)
                return { };

            definition.append(*component);

            skipExactly(buffer, '|');
            skipWhile<isCSSSpace>(buffer);
        }

        if (definition.isEmpty())
            return { };

        return CSSCustomPropertySyntax { definition };
    });
}

auto CSSCustomPropertySyntax::typeForTypeName(StringView dataTypeName) -> Type
{
    static constexpr std::pair<ComparableASCIILiteral, Type> mappings[] = {
        { "angle"_s, Type::Angle },
        { "color"_s, Type::Color },
        { "custom-ident"_s, Type::CustomIdent },
        { "image"_s, Type::Image },
        { "integer"_s, Type::Integer },
        { "length"_s, Type::Length },
        { "length-percentage"_s, Type::LengthPercentage },
        { "number"_s, Type::Number },
        { "percentage"_s, Type::Percentage },
        { "resolution"_s, Type::Resolution },
        { "string"_s, Type::String },
        { "time"_s, Type::Time },
        { "transform-function"_s, Type::TransformFunction },
        { "transform-list"_s, Type::TransformList },
        { "url"_s, Type::URL },
    };

    static constexpr SortedArrayMap typeMap { mappings };
    return typeMap.get(dataTypeName, Type::Unknown);
}

}

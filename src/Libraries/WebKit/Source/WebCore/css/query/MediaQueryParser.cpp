/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 23, 2024.
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
#include "MediaQueryParser.h"

#include "CSSMarkup.h"
#include "CSSTokenizer.h"
#include "CSSValueKeywords.h"
#include "CommonAtomStrings.h"
#include "GenericMediaQuerySerialization.h"
#include "MediaQueryFeatures.h"

namespace WebCore {
namespace MQ {

Vector<const FeatureSchema*> MediaQueryParser::featureSchemas()
{
    return Features::allSchemas();
}

MediaQueryList MediaQueryParser::parse(const String& string, const MediaQueryParserContext& context)
{
    auto tokenizer = CSSTokenizer::tryCreate(string);
    if (!tokenizer)
        return { };

    auto range = tokenizer->tokenRange();
    return parse(range, context);
}

MediaQueryList MediaQueryParser::parse(CSSParserTokenRange range, const MediaQueryParserContext& context)
{
    return consumeMediaQueryList(range, context);
}

std::optional<MediaQuery> MediaQueryParser::parseCondition(CSSParserTokenRange range, const MediaQueryParserContext& context)
{
    range.consumeWhitespace();

    if (range.atEnd())
        return MediaQuery { { }, allAtom() };

    auto condition = consumeCondition(range, context);
    if (!condition)
        return { };

    return MediaQuery { { }, { }, condition };
}

MediaQueryList MediaQueryParser::consumeMediaQueryList(CSSParserTokenRange& range, const MediaQueryParserContext& context)
{
    range.consumeWhitespace();

    if (range.atEnd())
        return { };

    MediaQueryList list;

    while (true) {
        auto begin = range.begin();
        while (!range.atEnd() && range.peek().type() != CommaToken)
            range.consumeComponentValue();

        auto subrange = range.makeSubRange(begin, &range.peek());

        auto consumeMediaQueryOrNotAll = [&] {
            if (auto query = consumeMediaQuery(subrange, context))
                return *query;
            // "A media query that does not match the grammar in the previous section must be replaced by not all during parsing."
            return MediaQuery { Prefix::Not, allAtom() };
        };

        list.append(consumeMediaQueryOrNotAll());

        if (range.atEnd())
            break;
        range.consumeIncludingWhitespace();
    }

    return list;
}

std::optional<MediaQuery> MediaQueryParser::consumeMediaQuery(CSSParserTokenRange& range, const MediaQueryParserContext& context)
{
    // <media-condition>

    auto rangeCopy = range;
    if (auto condition = consumeCondition(range, context)) {
        if (!range.atEnd())
            return { };
        return MediaQuery { { }, { }, condition };
    }

    range = rangeCopy;

    // [ not | only ]? <media-type> [ and <media-condition-without-or> ]

    auto consumePrefix = [&]() -> std::optional<Prefix> {
        if (range.peek().type() != IdentToken)
            return { };

        if (range.peek().id() == CSSValueNot) {
            range.consumeIncludingWhitespace();
            return Prefix::Not;
        }
        if (range.peek().id() == CSSValueOnly) {
            // 'only' doesn't do anything. It exists to hide the rule from legacy agents.
            range.consumeIncludingWhitespace();
            return Prefix::Only;
        }
        return { };
    };

    auto consumeMediaType = [&]() -> AtomString {
        if (range.peek().type() != IdentToken)
            return { };

        auto identifier = range.peek().id();
        if (identifier == CSSValueOnly || identifier == CSSValueNot || identifier == CSSValueAnd || identifier == CSSValueOr)
            return { };

        auto mediaType = range.consumeIncludingWhitespace().value().convertToASCIILowercaseAtom();
        if (mediaType == "layer"_s)
            return { };
        
        return mediaType;
    };

    auto prefix = consumePrefix();
    auto mediaType = consumeMediaType();

    if (mediaType.isNull())
        return { };

    if (range.atEnd())
        return MediaQuery { prefix, mediaType, { } };

    if (range.peek().type() != IdentToken || range.peek().id() != CSSValueAnd)
        return { };

    range.consumeIncludingWhitespace();

    auto condition = consumeCondition(range, context);
    if (!condition)
        return { };

    if (!range.atEnd())
        return { };

    if (condition->logicalOperator == LogicalOperator::Or)
        return { };

    return MediaQuery { prefix, mediaType, condition };
}

const FeatureSchema* MediaQueryParser::schemaForFeatureName(const AtomString& name, const MediaQueryParserContext& context, State& state)
{
    auto* schema = GenericMediaQueryParser<MediaQueryParser>::schemaForFeatureName(name, context, state);

    if (schema == &Features::prefersDarkInterface()) {
        if (!context.context.useSystemAppearance && !isUASheetBehavior(context.context.mode))
            return nullptr;
    }
    
    return schema;
}

const MediaProgressProviding* MediaQueryParser::mediaProgressProvidingSchemaForFeatureName(const AtomString& name, const MediaQueryParserContext&)
{
    using Map = MemoryCompactLookupOnlyRobinHoodHashMap<AtomString, const MediaProgressProviding*>;

    static NeverDestroyed<Map> schemas = [&] {
        Map map;
        for (auto& entry : Features::allMediaProgressProvidingSchemas())
            map.add(entry->name(), entry);
        return map;
    }();

    return schemas->get(name);
}

void serialize(StringBuilder& builder, const MediaQueryList& list)
{
    builder.append(interleave(list, serialize, ", "_s));
}

void serialize(StringBuilder& builder, const MediaQuery& query)
{
    if (query.prefix) {
        switch (*query.prefix) {
        case Prefix::Not:
            builder.append("not "_s);
            break;
        case Prefix::Only:
            builder.append("only "_s);
            break;
        }
    }

    if (!query.mediaType.isEmpty() && (!query.condition || query.prefix || query.mediaType != allAtom())) {
        serializeIdentifier(query.mediaType, builder);
        if (query.condition)
            builder.append(" and "_s);
    }

    if (query.condition)
        serialize(builder, *query.condition);
}

}
}

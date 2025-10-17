/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 29, 2021.
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

#include "CSSParserContext.h"
#include "CSSParserTokenRange.h"
#include "GenericMediaQueryTypes.h"
#include <wtf/RobinHoodHashMap.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {

struct MediaQueryParserContext;

namespace MQ {

struct FeatureParser {
    static std::optional<Feature> consumeFeature(CSSParserTokenRange&, const MediaQueryParserContext&);
    static std::optional<Feature> consumeBooleanOrPlainFeature(CSSParserTokenRange&, const MediaQueryParserContext&);
    static std::optional<Feature> consumeRangeFeature(CSSParserTokenRange&, const MediaQueryParserContext&);
    static RefPtr<CSSValue> consumeValue(CSSParserTokenRange&, const MediaQueryParserContext&);

    static bool validateFeatureAgainstSchema(Feature&, const FeatureSchema&);
};

template<typename ConcreteParser>
struct GenericMediaQueryParser  {
    struct State {
        std::optional<CSSValueID> inFunctionId;
    };
    static std::optional<Condition> consumeCondition(CSSParserTokenRange& range, const MediaQueryParserContext& context)
    {
        State state;
        return consumeCondition(range, context, state);
    }
    static std::optional<Condition> consumeCondition(CSSParserTokenRange&, const MediaQueryParserContext&, State&);
    static std::optional<QueryInParens> consumeQueryInParens(CSSParserTokenRange&, const MediaQueryParserContext&, State&);
    static std::optional<Feature> consumeAndValidateFeature(CSSParserTokenRange&, const MediaQueryParserContext&, State&);

    static bool isValidFunctionId(CSSValueID) { return false; }
    static const FeatureSchema* schemaForFeatureName(const AtomString&, const MediaQueryParserContext&, State&);
    static bool validateFeature(Feature&, const MediaQueryParserContext&, State&);
};

template<typename ConcreteParser>
std::optional<Condition> GenericMediaQueryParser<ConcreteParser>::consumeCondition(CSSParserTokenRange& range, const MediaQueryParserContext& context, State& state)
{
    if (range.peek().type() == IdentToken) {
        if (range.peek().id() == CSSValueNot) {
            range.consumeIncludingWhitespace();
            auto query = consumeQueryInParens(range, context, state);
            if (!query || !range.atEnd())
                return { };

            return Condition { LogicalOperator::Not, { *query } };
        }
    }

    Condition condition;

    auto consumeOperator = [&]() -> std::optional<LogicalOperator> {
        auto operatorToken = range.consumeIncludingWhitespace();
        if (operatorToken.type() != IdentToken)
            return { };
        if (operatorToken.id() == CSSValueAnd)
            return LogicalOperator::And;
        if (operatorToken.id() == CSSValueOr)
            return LogicalOperator::Or;
        return { };
    };

    do {
        if (!condition.queries.isEmpty()) {
            auto op = consumeOperator();
            if (!op)
                return { };
            if (condition.queries.size() > 1 && condition.logicalOperator != *op)
                return { };
            condition.logicalOperator = *op;
        }

        auto query = consumeQueryInParens(range, context, state);
        if (!query)
            return { };

        condition.queries.append(*query);
    } while (!range.atEnd());

    return condition;
}

template<typename ConcreteParser>
std::optional<QueryInParens> GenericMediaQueryParser<ConcreteParser>::consumeQueryInParens(CSSParserTokenRange& range, const MediaQueryParserContext& context, State& state)
{
    std::optional<CSSValueID> functionId;

    if (range.peek().type() == FunctionToken) {
        if (state.inFunctionId)
            return { };

        functionId = range.peek().functionId();
        if (!ConcreteParser::isValidFunctionId(*functionId)) {
            auto name = range.peek().value();
            auto functionRange = range.consumeBlock();
            range.consumeWhitespace();
            return GeneralEnclosed { name.toString(), functionRange.serialize() };
        }
    }

    if (!functionId && range.peek().type() != LeftParenthesisToken)
        return { };

    auto originalBlockRange = range.consumeBlock();
    range.consumeWhitespace();

    auto blockRange = originalBlockRange;
    blockRange.consumeWhitespace();

    SetForScope functionScope(state.inFunctionId, functionId ? *functionId : state.inFunctionId);

    auto conditionRange = blockRange;
    if (auto condition = consumeCondition(conditionRange, context, state)) {
        condition->functionId = functionId;
        return { condition };
    }

    auto featureRange = blockRange;
    if (auto feature = consumeAndValidateFeature(featureRange, context, state)) {
        feature->functionId = functionId;
        return { *feature };
    }

    return GeneralEnclosed { functionId ? nameString(*functionId) : nullAtom(), originalBlockRange.serialize() };
}

template<typename ConcreteParser>
std::optional<Feature> GenericMediaQueryParser<ConcreteParser>::consumeAndValidateFeature(CSSParserTokenRange& range, const MediaQueryParserContext& context, State& state)
{
    auto feature = FeatureParser::consumeFeature(range, context);
    if (!feature)
        return { };

    if (!validateFeature(*feature, context, state))
        return { };

    return feature;
}

template<typename ConcreteParser>
bool GenericMediaQueryParser<ConcreteParser>::validateFeature(Feature& feature, const MediaQueryParserContext& context, State& state)
{
    auto* schema = ConcreteParser::schemaForFeatureName(feature.name, context, state);
    if (!schema)
        return false;
    return FeatureParser::validateFeatureAgainstSchema(feature, *schema);
}

template<typename ConcreteParser>
const FeatureSchema* GenericMediaQueryParser<ConcreteParser>::schemaForFeatureName(const AtomString& name, const MediaQueryParserContext&, State&)
{
    using SchemaMap = MemoryCompactLookupOnlyRobinHoodHashMap<AtomString, const FeatureSchema*>;

    static NeverDestroyed<SchemaMap> schemas = [&] {
        SchemaMap map;
        for (auto& entry : ConcreteParser::featureSchemas())
            map.add(entry->name, entry);
        return map;
    }();

    return schemas->get(name);
}

}
}

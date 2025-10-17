/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 24, 2024.
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
#include "ContainerQueryParser.h"

#include "CSSPrimitiveValue.h"
#include "CSSPropertyParser.h"
#include "CSSPropertyParserConsumer+Conditional.h"
#include "ContainerQueryFeatures.h"
#include "MediaQueryParserContext.h"

namespace WebCore {
namespace CQ {

Vector<const MQ::FeatureSchema*> ContainerQueryParser::featureSchemas()
{
    return Features::allSchemas();
}

std::optional<ContainerQuery> ContainerQueryParser::consumeContainerQuery(CSSParserTokenRange& range, const MediaQueryParserContext& context)
{
    auto consumeName = [&] {
        if (range.peek().type() == LeftParenthesisToken || range.peek().type() == FunctionToken)
            return nullAtom();
        auto nameValue = CSSPropertyParserHelpers::consumeSingleContainerName(range, context.context);
        if (!nameValue)
            return nullAtom();
        return AtomString { nameValue->stringValue() };
    };

    auto name = consumeName();

    auto condition = consumeCondition(range, context);
    if (!condition)
        return { };

    OptionSet<Axis> requiredAxes;
    auto containsUnknownFeature = ContainsUnknownFeature::No;

    traverseFeatures(*condition, [&](auto& feature) {
        requiredAxes.add(requiredAxesForFeature(feature));
        if (!feature.schema)
            containsUnknownFeature = ContainsUnknownFeature::Yes;
    });

    return ContainerQuery { name, *condition, requiredAxes, containsUnknownFeature };
}

bool ContainerQueryParser::isValidFunctionId(CSSValueID functionId)
{
    return functionId == CSSValueStyle;
}

const MQ::FeatureSchema* ContainerQueryParser::schemaForFeatureName(const AtomString& name, const MediaQueryParserContext& context, State& state)
{
    if (state.inFunctionId == CSSValueStyle && context.context.cssStyleQueriesEnabled)
        return &Features::style();

    return GenericMediaQueryParser<ContainerQueryParser>::schemaForFeatureName(name, context, state);
}

const ContainerProgressProviding* ContainerQueryParser::containerProgressProvidingSchemaForFeatureName(const AtomString& name, const MediaQueryParserContext&)
{
    using Map = MemoryCompactLookupOnlyRobinHoodHashMap<AtomString, const ContainerProgressProviding*>;

    static NeverDestroyed<Map> schemas = [&] {
        Map map;
        for (auto& entry : Features::allContainerProgressProvidingSchemas())
            map.add(entry->name(), entry);
        return map;
    }();

    return schemas->get(name);
}

}
}

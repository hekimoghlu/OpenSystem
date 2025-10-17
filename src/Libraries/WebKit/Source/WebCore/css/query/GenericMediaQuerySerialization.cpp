/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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
#include "GenericMediaQuerySerialization.h"

#include "CSSMarkup.h"
#include "CSSValue.h"

namespace WebCore {
namespace MQ {

static void serialize(StringBuilder& builder, const QueryInParens& queryInParens)
{
    WTF::switchOn(queryInParens, [&](auto& node) {
        if (node.functionId)
            builder.append(nameString(*node.functionId));
        builder.append('(');
        serialize(builder, node);
        builder.append(')');
    }, [&](const GeneralEnclosed& generalEnclosed) {
        builder.append(generalEnclosed.name);
        builder.append('(');
        builder.append(generalEnclosed.text);
        builder.append(')');
    });
}

void serialize(StringBuilder& builder, const Condition& condition)
{
    if (condition.queries.size() == 1 && condition.logicalOperator == LogicalOperator::Not) {
        builder.append("not "_s);
        serialize(builder, condition.queries.first());
        return;
    }

    for (auto& query : condition.queries) {
        if (&query != &condition.queries.first())
            builder.append(condition.logicalOperator == LogicalOperator::And ? " and "_s : " or "_s);
        serialize(builder, query);
    }
}

void serialize(StringBuilder& builder, const Feature& feature)
{
    auto serializeRangeComparisonOperator = [&](ComparisonOperator op) {
        builder.append(' ');
        switch (op) {
        case ComparisonOperator::LessThan:
            builder.append('<');
            break;
        case ComparisonOperator::LessThanOrEqual:
            builder.append("<="_s);
            break;
        case ComparisonOperator::Equal:
            builder.append('=');
            break;
        case ComparisonOperator::GreaterThan:
            builder.append('>');
            break;
        case ComparisonOperator::GreaterThanOrEqual:
            builder.append(">="_s);
            break;
        }
        builder.append(' ');
    };

    switch (feature.syntax) {
    case Syntax::Boolean:
        serializeIdentifier(feature.name, builder);
        break;

    case Syntax::Plain:
        switch (feature.rightComparison->op) {
        case MQ::ComparisonOperator::LessThanOrEqual:
            builder.append("max-"_s);
            break;
        case MQ::ComparisonOperator::Equal:
            break;
        case MQ::ComparisonOperator::GreaterThanOrEqual:
            builder.append("min-"_s);
            break;
        case MQ::ComparisonOperator::LessThan:
        case MQ::ComparisonOperator::GreaterThan:
            ASSERT_NOT_REACHED();
            break;
        }
        serializeIdentifier(feature.name, builder);

        builder.append(": "_s, feature.rightComparison->value->cssText());
        break;

    case Syntax::Range:
        if (feature.leftComparison) {
            builder.append(feature.leftComparison->value->cssText());
            serializeRangeComparisonOperator(feature.leftComparison->op);
        }

        serializeIdentifier(feature.name, builder);

        if (feature.rightComparison) {
            serializeRangeComparisonOperator(feature.rightComparison->op);
            builder.append(feature.rightComparison->value->cssText());
        }
        break;
    }
}

}
}

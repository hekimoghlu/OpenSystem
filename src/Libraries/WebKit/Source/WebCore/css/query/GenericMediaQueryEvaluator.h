/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 25, 2025.
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

#include "CSSToLengthConversionData.h"
#include "CSSValueKeywords.h"
#include "GenericMediaQueryTypes.h"
#include "LayoutUnit.h"

namespace WebCore {

class RenderElement;

namespace MQ {

EvaluationResult evaluateLengthFeature(const Feature&, LayoutUnit, const CSSToLengthConversionData&);
EvaluationResult evaluateRatioFeature(const Feature&, FloatSize, const CSSToLengthConversionData&);
EvaluationResult evaluateBooleanFeature(const Feature&, bool, const CSSToLengthConversionData&);
EvaluationResult evaluateIntegerFeature(const Feature&, int, const CSSToLengthConversionData&);
EvaluationResult evaluateNumberFeature(const Feature&, double, const CSSToLengthConversionData&);
EvaluationResult evaluateResolutionFeature(const Feature&, float, const CSSToLengthConversionData&);
EvaluationResult evaluateIdentifierFeature(const Feature&, CSSValueID, const CSSToLengthConversionData&);

template<typename ConcreteEvaluator>
class GenericMediaQueryEvaluator {
public:
    EvaluationResult evaluateQueryInParens(const QueryInParens&, const FeatureEvaluationContext&) const;
    EvaluationResult evaluateCondition(const Condition&, const FeatureEvaluationContext&) const;
    EvaluationResult evaluateFeature(const Feature&, const FeatureEvaluationContext&) const;

private:
    const ConcreteEvaluator& concreteEvaluator() const { return static_cast<const ConcreteEvaluator&>(*this); }
};

template<typename ConcreteEvaluator>
EvaluationResult GenericMediaQueryEvaluator<ConcreteEvaluator>::evaluateQueryInParens(const QueryInParens& queryInParens, const FeatureEvaluationContext& context) const
{
    return WTF::switchOn(queryInParens, [&](const Condition& condition) {
        return evaluateCondition(condition, context);
    }, [&](const MQ::Feature& feature) {
        return concreteEvaluator().evaluateFeature(feature, context);
    }, [&](const MQ::GeneralEnclosed&) {
        return MQ::EvaluationResult::Unknown;
    });
}

template<typename ConcreteEvaluator>
EvaluationResult GenericMediaQueryEvaluator<ConcreteEvaluator>::evaluateCondition(const Condition& condition, const FeatureEvaluationContext& context) const
{
    if (condition.queries.isEmpty())
        return EvaluationResult::Unknown;

    switch (condition.logicalOperator) {
    case LogicalOperator::Not:
        return !concreteEvaluator().evaluateQueryInParens(condition.queries.first(), context);

    // Kleene 3-valued logic.
    case LogicalOperator::And: {
        auto result = EvaluationResult::True;
        for (auto& query : condition.queries) {
            auto queryResult = concreteEvaluator().evaluateQueryInParens(query, context);
            if (queryResult == EvaluationResult::False)
                return EvaluationResult::False;
            if (queryResult == EvaluationResult::Unknown)
                result = EvaluationResult::Unknown;
        }
        return result;
    }

    case LogicalOperator::Or: {
        auto result = EvaluationResult::False;
        for (auto& query : condition.queries) {
            auto queryResult = concreteEvaluator().evaluateQueryInParens(query, context);
            if (queryResult == EvaluationResult::True)
                return EvaluationResult::True;
            if (queryResult == EvaluationResult::Unknown)
                result = EvaluationResult::Unknown;
        }
        return result;
    }
    }
    RELEASE_ASSERT_NOT_REACHED();
}

template<typename ConcreteEvaluator>
EvaluationResult GenericMediaQueryEvaluator<ConcreteEvaluator>::evaluateFeature(const Feature& feature, const FeatureEvaluationContext& context) const
{
    if (!feature.schema)
        return MQ::EvaluationResult::Unknown;

    return feature.schema->evaluate(feature, context);
}

inline EvaluationResult operator&(EvaluationResult left, EvaluationResult right)
{
    if (left == EvaluationResult::Unknown || right == EvaluationResult::Unknown)
        return EvaluationResult::Unknown;
    if (left == EvaluationResult::True && right == EvaluationResult::True)
        return EvaluationResult::True;
    return EvaluationResult::False;
}

inline EvaluationResult operator!(EvaluationResult result)
{
    switch (result) {
    case EvaluationResult::True:
        return EvaluationResult::False;
    case EvaluationResult::False:
        return EvaluationResult::True;
    case EvaluationResult::Unknown:
        return EvaluationResult::Unknown;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

inline EvaluationResult toEvaluationResult(bool boolean)
{
    return boolean ? EvaluationResult::True : EvaluationResult::False;
}

}
}

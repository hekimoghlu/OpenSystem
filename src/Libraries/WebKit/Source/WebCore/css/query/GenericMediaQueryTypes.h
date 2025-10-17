/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 25, 2021.
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
#include "CSSValue.h"
#include "CSSValueKeywords.h"
#include <wtf/CheckedPtr.h>
#include <wtf/OptionSet.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

class RenderElement;

namespace MQ {

enum class LogicalOperator : uint8_t { And, Or, Not };
enum class ComparisonOperator : uint8_t { LessThan, LessThanOrEqual, Equal, GreaterThan, GreaterThanOrEqual };
enum class Syntax : uint8_t { Boolean, Plain, Range };

struct Condition;
struct FeatureSchema;

struct Comparison {
    ComparisonOperator op;
    RefPtr<CSSValue> value;
};

struct Feature {
    AtomString name;
    Syntax syntax;
    std::optional<Comparison> leftComparison;
    std::optional<Comparison> rightComparison;

    std::optional<CSSValueID> functionId { };

    const FeatureSchema* schema { nullptr };
};

struct GeneralEnclosed {
    String name;
    String text;
};

using QueryInParens = std::variant<Condition, Feature, GeneralEnclosed>;

struct Condition {
    LogicalOperator logicalOperator { LogicalOperator::And };
    Vector<QueryInParens> queries;

    std::optional<CSSValueID> functionId { };
};

enum class EvaluationResult : uint8_t { False, True, Unknown };

enum class MediaQueryDynamicDependency : uint8_t  {
    Viewport = 1 << 0,
    Appearance = 1 << 1,
    Accessibility = 1 << 2,
};

struct FeatureEvaluationContext {
    WeakRef<const Document, WeakPtrImplWithEventTargetData> document;
    CSSToLengthConversionData conversionData { };
    CheckedPtr<const RenderElement> renderer { };
};

struct FeatureSchema {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    enum class Type : uint8_t { Discrete, Range };
    enum class ValueType : uint8_t { Integer, Number, Length, Ratio, Resolution, Identifier, CustomProperty };

    AtomString name;
    Type type;
    ValueType valueType;
    OptionSet<MediaQueryDynamicDependency> dependencies;
    FixedVector<CSSValueID> valueIdentifiers;

    virtual EvaluationResult evaluate(const Feature&, const FeatureEvaluationContext&) const { return EvaluationResult::Unknown; }

    FeatureSchema(const AtomString& name, Type type, ValueType valueType, OptionSet<MediaQueryDynamicDependency> dependencies, FixedVector<CSSValueID>&& valueIdentifiers = { })
        : name(name)
        , type(type)
        , valueType(valueType)
        , dependencies(dependencies)
        , valueIdentifiers(WTFMove(valueIdentifiers))
    { }
    virtual ~FeatureSchema() = default;
};

template<typename TraverseFunction> void traverseFeatures(const Condition&, TraverseFunction&&);

template<typename TraverseFunction>
void traverseFeatures(const QueryInParens& queryInParens, TraverseFunction&& function)
{
    return WTF::switchOn(queryInParens, [&](const Condition& condition) {
        traverseFeatures(condition, function);
    }, [&](const MQ::Feature& feature) {
        function(feature);
    }, [&](const MQ::GeneralEnclosed&) {
        MQ::Feature dummy { };
        function(dummy);
    });
}

template<typename TraverseFunction>
void traverseFeatures(const Condition& condition, TraverseFunction&& function)
{
    for (auto& queryInParens : condition.queries)
        traverseFeatures(queryInParens, function);
}


}
}

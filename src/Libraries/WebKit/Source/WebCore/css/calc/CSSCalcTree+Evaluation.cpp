/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 21, 2022.
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
#include "CSSCalcTree+Evaluation.h"

#include "AnchorPositionEvaluator.h"
#include "CSSCalcSymbolTable.h"
#include "CSSCalcTree+ContainerProgressEvaluator.h"
#include "CSSCalcTree+Mappings.h"
#include "CSSCalcTree+MediaProgressEvaluator.h"
#include "CSSCalcTree+Simplification.h"
#include "CSSCalcTree.h"
#include "CalculationExecutor.h"
#include "RenderStyle.h"
#include "RenderStyleInlines.h"
#include "StyleBuilderState.h"

namespace WebCore {
namespace CSSCalc {

static auto evaluate(const CSS::Keyword::None&, const EvaluationOptions&) -> std::optional<Calculation::None>;
static auto evaluate(const ChildOrNone&, const EvaluationOptions&) -> std::optional<std::variant<double, Calculation::None>>;
static auto evaluate(const std::optional<Child>&, const EvaluationOptions&) -> std::optional<std::optional<double>>;
static auto evaluate(const Child&, const EvaluationOptions&) -> std::optional<double>;
static auto evaluate(const Number&, const EvaluationOptions&) -> std::optional<double>;
static auto evaluate(const Percentage&, const EvaluationOptions&) -> std::optional<double>;
static auto evaluate(const CanonicalDimension&, const EvaluationOptions&) -> std::optional<double>;
static auto evaluate(const NonCanonicalDimension&, const EvaluationOptions&) -> std::optional<double>;
static auto evaluate(const Symbol&, const EvaluationOptions&) -> std::optional<double>;
static auto evaluate(const IndirectNode<Sum>&, const EvaluationOptions&) -> std::optional<double>;
static auto evaluate(const IndirectNode<Product>&, const EvaluationOptions&) -> std::optional<double>;
static auto evaluate(const IndirectNode<Min>&, const EvaluationOptions&) -> std::optional<double>;
static auto evaluate(const IndirectNode<Max>&, const EvaluationOptions&) -> std::optional<double>;
static auto evaluate(const IndirectNode<Hypot>&, const EvaluationOptions&) -> std::optional<double>;
static auto evaluate(const IndirectNode<Random>&, const EvaluationOptions&) -> std::optional<double>;
static auto evaluate(const IndirectNode<MediaProgress>&, const EvaluationOptions&) -> std::optional<double>;
static auto evaluate(const IndirectNode<ContainerProgress>&, const EvaluationOptions&) -> std::optional<double>;
static auto evaluate(const IndirectNode<Anchor>&, const EvaluationOptions&) -> std::optional<double>;
static auto evaluate(const IndirectNode<AnchorSize>&, const EvaluationOptions&) -> std::optional<double>;
template<typename Op>
static auto evaluate(const IndirectNode<Op>&, const EvaluationOptions&) -> std::optional<double>;

// MARK: Evaluation.

template<typename Op, typename... Args> static std::optional<double> executeMathOperationAfterUnwrapping(Args&&... args)
{
    if ((!args.has_value() || ...))
        return std::nullopt;

    return Calculation::executeOperation<ToCalculationTreeOp<Op>>(args.value()...);
}

template<typename Op> static std::optional<double> executeVariadicMathOperationAfterUnwrapping(const IndirectNode<Op>& op, const EvaluationOptions& options)
{
    bool failure = false;
    auto result = Calculation::executeOperation<ToCalculationTreeOp<Op>>(op->children.value, [&](const auto& child) -> double {
        if (auto value = evaluate(child, options))
            return *value;
        failure = true;
        return std::numeric_limits<double>::quiet_NaN();
    });

    if (failure)
        return std::nullopt;

    return result;
}

std::optional<Calculation::None> evaluate(const CSS::Keyword::None&, const EvaluationOptions&)
{
    return Calculation::None { };
}

std::optional<std::variant<double, Calculation::None>> evaluate(const ChildOrNone& root, const EvaluationOptions& options)
{
    return WTF::switchOn(root,
        [&](const auto& root) -> std::optional<std::variant<double, Calculation::None>> {
            if (auto value = evaluate(root, options))
                return std::variant<double, Calculation::None> { *value };
            return std::nullopt;
        }
    );
}

std::optional<double> evaluate(const Child& root, const EvaluationOptions& options)
{
    return WTF::switchOn(root, [&](const auto& root) { return evaluate(root, options); });
}

std::optional<std::optional<double>> evaluate(const std::optional<Child>& root, const EvaluationOptions& options)
{
    if (root)
        return std::optional<double> { evaluate(*root, options) };
    return std::optional<double> { std::nullopt };
}

std::optional<double> evaluate(const Number& number, const EvaluationOptions&)
{
    return number.value;
}

std::optional<double> evaluate(const Percentage& percentage, const EvaluationOptions&)
{
    return percentage.value;
}

std::optional<double> evaluate(const CanonicalDimension& root, const EvaluationOptions&)
{
    return root.value;
}

std::optional<double> evaluate(const NonCanonicalDimension& root, const EvaluationOptions& options)
{
    if (auto canonical = canonicalize(root, options.conversionData))
        return evaluate(*canonical, options);

    return std::nullopt;
}

std::optional<double> evaluate(const Symbol& root, const EvaluationOptions& options)
{
    if (auto value = options.symbolTable.get(root.id))
        return evaluate(makeNumeric(value->value, root.unit), options);

    ASSERT_NOT_REACHED();
    return std::nullopt;
}

std::optional<double> evaluate(const IndirectNode<Sum>& root, const EvaluationOptions& options)
{
    return executeVariadicMathOperationAfterUnwrapping(root, options);
}

std::optional<double> evaluate(const IndirectNode<Product>& root, const EvaluationOptions& options)
{
    return executeVariadicMathOperationAfterUnwrapping(root, options);
}

std::optional<double> evaluate(const IndirectNode<Min>& root, const EvaluationOptions& options)
{
    return executeVariadicMathOperationAfterUnwrapping(root, options);
}

std::optional<double> evaluate(const IndirectNode<Max>& root, const EvaluationOptions& options)
{
    return executeVariadicMathOperationAfterUnwrapping(root, options);
}

std::optional<double> evaluate(const IndirectNode<Hypot>& root, const EvaluationOptions& options)
{
    return executeVariadicMathOperationAfterUnwrapping(root, options);
}

std::optional<double> evaluate(const IndirectNode<Random>& root, const EvaluationOptions& options)
{
    if (!options.conversionData || !options.conversionData->styleBuilderState())
        return { };
    if (root->cachingOptions.perElement && !options.conversionData->styleBuilderState()->element())
        return { };

    auto min = evaluate(root->min, options);
    if (!min)
        return { };

    auto max = evaluate(root->max, options);
    if (!min)
        return { };

    auto step = evaluate(root->step, options);
    if (!step)
        return { };

    // RandomKeyMap relies on using NaN for HashTable deleted/empty values but
    // the result is always NaN if either is NaN, so we can return early here.
    if (std::isnan(*min) || std::isnan(*max))
        return std::numeric_limits<double>::quiet_NaN();

    auto keyMap = options.conversionData->styleBuilderState()->randomKeyMap(
        root->cachingOptions.perElement
    );

    auto randomUnitInterval = keyMap->lookupUnitInterval(
        root->cachingOptions.identifier,
        *min,
        *max,
        *step
    );

    return Calculation::executeOperation<ToCalculationTreeOp<Random>>(randomUnitInterval, *min, *max, *step);
}

std::optional<double> evaluate(const IndirectNode<MediaProgress>& root, const EvaluationOptions& options)
{
    if (!options.conversionData || !options.conversionData->styleBuilderState())
        return { };

    auto start = evaluate(root->start, options);
    if (!start)
        return { };

    auto end = evaluate(root->end, options);
    if (!end)
        return { };

    Ref document = options.conversionData->styleBuilderState()->document();
    auto value = evaluateMediaProgress(root, document, *options.conversionData);
    return Calculation::executeOperation<ToCalculationTreeOp<Progress>>(value, *start, *end);
}

std::optional<double> evaluate(const IndirectNode<ContainerProgress>& root, const EvaluationOptions& options)
{
    if (!options.conversionData || !options.conversionData->styleBuilderState() || !options.conversionData->styleBuilderState()->element())
        return { };

    auto start = evaluate(root->start, options);
    if (!start)
        return { };

    auto end = evaluate(root->end, options);
    if (!end)
        return { };

    Ref element = *options.conversionData->styleBuilderState()->element();
    auto value = evaluateContainerProgress(root, element, *options.conversionData);
    if (!value)
        return { };

    return Calculation::executeOperation<ToCalculationTreeOp<Progress>>(*value, *start, *end);
}

std::optional<double> evaluate(const IndirectNode<Anchor>& anchor, const EvaluationOptions& options)
{
    if (!options.conversionData || !options.conversionData->styleBuilderState())
        return { };

    auto result = evaluateWithoutFallback(*anchor, options);

    // https://drafts.csswg.org/css-anchor-position-1/#anchor-valid
    // "If any of these conditions are false, the anchor() function resolves to its specified fallback value.
    // If no fallback value is specified, it makes the declaration referencing it invalid at computed-value time."
    if (!result && anchor->fallback)
        result = evaluate(*anchor->fallback, options);

    if (!result)
        options.conversionData->styleBuilderState()->setCurrentPropertyInvalidAtComputedValueTime();

    return result;
}

std::optional<double> evaluate(const IndirectNode<AnchorSize>& anchorSize, const EvaluationOptions& options)
{
    if (!options.conversionData || !options.conversionData->styleBuilderState())
        return { };

    auto& builderState = *options.conversionData->styleBuilderState();

    std::optional<Style::ScopedName> anchorSizeScopedName;
    if (!anchorSize->elementName.isNull()) {
        anchorSizeScopedName = Style::ScopedName {
            .name = anchorSize->elementName,
            .scopeOrdinal = builderState.styleScopeOrdinal()
        };
    }

    auto result = Style::AnchorPositionEvaluator::evaluateSize(builderState, anchorSizeScopedName, anchorSize->dimension);

    if (!result && anchorSize->fallback)
        result = evaluate(*anchorSize->fallback, options);

    if (!result)
        options.conversionData->styleBuilderState()->setCurrentPropertyInvalidAtComputedValueTime();

    return result;
}

template<typename Op> std::optional<double> evaluate(const IndirectNode<Op>& root, const EvaluationOptions& options)
{
    return WTF::apply([&](const auto& ...x) { return executeMathOperationAfterUnwrapping<Op>(evaluate(x, options)...); } , *root);
}

std::optional<double> evaluateDouble(const Tree& tree, const EvaluationOptions& options)
{
    return evaluate(tree.root, options);
}

std::optional<double> evaluateWithoutFallback(const Anchor& anchor, const EvaluationOptions& options)
{
    auto& builderState = *options.conversionData->styleBuilderState();

    auto side = WTF::switchOn(anchor.side,
        [&](const Child& percentage) -> Style::AnchorPositionEvaluator::Side {
            return evaluate(percentage, options).value_or(0) / 100;
        }, [&](CSSValueID sideID) -> Style::AnchorPositionEvaluator::Side {
            return sideID;
        }
    );

    std::optional<Style::ScopedName> anchorScopedName;
    if (!anchor.elementName.isNull()) {
        anchorScopedName = Style::ScopedName {
            .name = anchor.elementName,
            .scopeOrdinal = builderState.styleScopeOrdinal()
        };
    }

    return Style::AnchorPositionEvaluator::evaluate(builderState, anchorScopedName, side);
}

} // namespace CSSCalc
} // namespace WebCore

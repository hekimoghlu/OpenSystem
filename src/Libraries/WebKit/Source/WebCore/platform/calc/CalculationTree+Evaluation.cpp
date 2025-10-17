/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 13, 2025.
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
#include "CalculationTree+Evaluation.h"

#include "CalculationExecutor.h"
#include "CalculationTree.h"
#include <wtf/StdLibExtras.h>

namespace WebCore {
namespace Calculation {

static auto evaluate(const None&, double percentResolutionLength) -> None;
static auto evaluate(const ChildOrNone&, double percentResolutionLength) -> std::variant<double, None>;
static auto evaluate(const std::optional<Child>&, double percentResolutionLength) -> std::optional<double>;
static auto evaluate(const Child&, double percentResolutionLength) -> double;
static auto evaluate(const Number&, double percentResolutionLength) -> double;
static auto evaluate(const Percentage&, double percentResolutionLength) -> double;
static auto evaluate(const Dimension&, double percentResolutionLength) -> double;
static auto evaluate(const IndirectNode<Sum>&, double percentResolutionLength) -> double;
static auto evaluate(const IndirectNode<Product>&, double percentResolutionLength) -> double;
static auto evaluate(const IndirectNode<Min>&, double percentResolutionLength) -> double;
static auto evaluate(const IndirectNode<Max>&, double percentResolutionLength) -> double;
static auto evaluate(const IndirectNode<Hypot>&, double percentResolutionLength) -> double;
static auto evaluate(const IndirectNode<Random>&, double percentResolutionLength) -> double;
static auto evaluate(const IndirectNode<Blend>&, double percentResolutionLength) -> double;
template<typename Op>
static auto evaluate(const IndirectNode<Op>&, double percentResolutionLength) -> double;

// MARK: Evaluation.

None evaluate(const None& root, double)
{
    return root;
}

std::variant<double, None> evaluate(const ChildOrNone& root, double percentResolutionLength)
{
    return WTF::switchOn(root, [&](const auto& root) { return std::variant<double, None> { evaluate(root, percentResolutionLength) }; });
}

double evaluate(const Child& root, double percentResolutionLength)
{
    return WTF::switchOn(root, [&](const auto& root) { return evaluate(root, percentResolutionLength); });
}

std::optional<double> evaluate(const std::optional<Child>& root, double percentResolutionLength)
{
    if (root)
        return static_cast<double>(evaluate(*root, percentResolutionLength));
    return std::nullopt;
}

double evaluate(const Number& number, double)
{
    return number.value;
}

double evaluate(const Percentage& percentage, double percentResolutionLength)
{
    return percentResolutionLength * percentage.value / 100.0;
}

double evaluate(const Dimension& root, double)
{
    return root.value;
}

double evaluate(const IndirectNode<Sum>& root, double percentResolutionLength)
{
    return executeOperation<Sum>(root->children.value, [&](const auto& child) -> double {
        return evaluate(child, percentResolutionLength);
    });
}

double evaluate(const IndirectNode<Product>& root, double percentResolutionLength)
{
    return executeOperation<Product>(root->children.value, [&](const auto& child) -> double {
        return evaluate(child, percentResolutionLength);
    });
}

double evaluate(const IndirectNode<Min>& root, double percentResolutionLength)
{
    return executeOperation<Min>(root->children.value, [&](const auto& child) -> double {
        return evaluate(child, percentResolutionLength);
    });
}

double evaluate(const IndirectNode<Max>& root, double percentResolutionLength)
{
    return executeOperation<Max>(root->children.value, [&](const auto& child) -> double {
        return evaluate(child, percentResolutionLength);
    });
}

double evaluate(const IndirectNode<Hypot>& root, double percentResolutionLength)
{
    return executeOperation<Hypot>(root->children.value, [&](const auto& child) -> double {
        return evaluate(child, percentResolutionLength);
    });
}

double evaluate(const IndirectNode<Random>& root, double percentResolutionLength)
{
    auto min = evaluate(root->min, percentResolutionLength);
    auto max = evaluate(root->max, percentResolutionLength);
    auto step = evaluate(root->step, percentResolutionLength);

    // RandomKeyMap relies on using NaN for HashTable deleted/empty values but
    // the result is always NaN if either is NaN, so we can return early here.
    if (std::isnan(min) || std::isnan(max))
        return std::numeric_limits<double>::quiet_NaN();

    Ref keyMap = root->cachingOptions.keyMap;
    auto randomUnitInterval = keyMap->lookupUnitInterval(
        root->cachingOptions.identifier,
        min,
        max,
        step
    );

    return executeOperation<Random>(randomUnitInterval, min, max, step);
}

double evaluate(const IndirectNode<Blend>& root, double percentResolutionLength)
{
    return (1.0 - root->progress) * evaluate(root->from, percentResolutionLength) + root->progress * evaluate(root->to, percentResolutionLength);
}

template<typename Op> double evaluate(const IndirectNode<Op>& root, double percentResolutionLength)
{
    return WTF::apply([&](const auto& ...x) { return executeOperation<Op>(evaluate(x, percentResolutionLength)...); } , *root);
}

double evaluate(const Tree& tree, double percentResolutionLength)
{
    return evaluate(tree.root, percentResolutionLength);
}

} // namespace Calculation
} // namespace WebCore

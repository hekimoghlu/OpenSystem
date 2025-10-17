/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 24, 2025.
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
#include "CalculationTree+Copy.h"

#include "CalculationTree.h"
#include <wtf/StdLibExtras.h>

namespace WebCore {
namespace Calculation {

static auto copy(double) -> double;
static auto copy(const std::optional<Child>& root) -> std::optional<Child>;
static auto copy(const Random::CachingOptions&) -> Random::CachingOptions;
static auto copy(const None&) -> None;
static auto copy(const ChildOrNone&) -> ChildOrNone;
static auto copy(const Children&) -> Children;
static auto copy(const Child&) -> Child;
template<Leaf Op>
Child copy(const Op&);
template<typename Op>
static auto copy(const IndirectNode<Op>&) -> Child;

// MARK: Copying

Tree copy(const Tree& tree)
{
    return Tree { .root = copy(tree.root) };
}

double copy(double value)
{
    return value;
}

std::optional<Child> copy(const std::optional<Child>& root)
{
    if (root)
        return copy(*root);
    return std::nullopt;
}

Random::CachingOptions copy(const Random::CachingOptions& options)
{
    return options;
}

None copy(const None& none)
{
    return none;
}

ChildOrNone copy(const ChildOrNone& root)
{
    return WTF::switchOn(root, [&](const auto& root) { return ChildOrNone { copy(root) }; });
}

Children copy(const Children& children)
{
    return WTF::map(children, [&](const auto& child) { return copy(child); });
}

Child copy(const Child& root)
{
    return WTF::switchOn(root, [&](const auto& root) { return copy(root); });
}

template<Leaf Op> Child copy(const Op& root)
{
    return { root };
}

template<typename Op> Child copy(const IndirectNode<Op>& root)
{
    return makeChild(WTF::apply([](const auto& ...x) { return Op { copy(x)... }; } , *root));
}

} // namespace Calculation
} // namespace WebCore

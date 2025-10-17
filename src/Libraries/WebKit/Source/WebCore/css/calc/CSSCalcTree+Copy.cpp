/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 18, 2025.
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
#include "CSSCalcTree+Copy.h"

#include "CSSCalcTree.h"

namespace WebCore {
namespace CSSCalc {

static auto copy(const MQ::MediaProgressProviding*) -> const MQ::MediaProgressProviding*;
static auto copy(const CQ::ContainerProgressProviding*) -> const CQ::ContainerProgressProviding*;
static auto copy(const Random::CachingOptions&) -> Random::CachingOptions;
static auto copy(const CSSValueID&) -> CSSValueID;
static auto copy(const AtomString&) -> AtomString;
static auto copy(const CSS::Keyword::None&) -> CSS::Keyword::None;
static auto copy(const std::optional<Child>& root) -> std::optional<Child>;
static auto copy(const ChildOrNone&) -> ChildOrNone;
static auto copy(const Children&) -> Children;
static auto copy(const Child&) -> Child;
template<Leaf Op> Child copy(const Op&);
template<typename Op> static auto copy(const IndirectNode<Op>&) -> Child;
static auto copy(const IndirectNode<Anchor>&) -> Child;
static auto copy(const IndirectNode<AnchorSize>&) -> Child;

// MARK: Copying

const MQ::MediaProgressProviding* copy(const MQ::MediaProgressProviding* root)
{
    return root;
}

const CQ::ContainerProgressProviding* copy(const CQ::ContainerProgressProviding* root)
{
    return root;
}

Random::CachingOptions copy(const Random::CachingOptions& root)
{
    return root;
}

CSSValueID copy(const CSSValueID& root)
{
    return root;
}

AtomString copy(const AtomString& root)
{
    return root;
}

CSS::Keyword::None copy(const CSS::Keyword::None& root)
{
    return root;
}

std::optional<Child> copy(const std::optional<Child>& root)
{
    if (root)
        return copy(*root);
    return std::nullopt;
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
    return makeChild(WTF::apply([](const auto& ...x) { return Op { copy(x)... }; } , *root), root.type);
}

AnchorSide copy(const AnchorSide& root)
{
    return WTF::switchOn(root, [&](const auto& root) { return AnchorSide { copy(root) }; });
}

Child copy(const IndirectNode<Anchor>& anchor)
{
    return makeChild(Anchor { .elementName = anchor->elementName, .side = copy(anchor->side), .fallback = copy(anchor->fallback) }, anchor.type);
}

Child copy(const IndirectNode<AnchorSize>& anchorSize)
{
    AnchorSize copyAnchorSize {
        .elementName = anchorSize->elementName,
        .dimension = anchorSize->dimension,
        .fallback = copy(anchorSize->fallback)
    };

    return makeChild(WTFMove(copyAnchorSize), anchorSize.type);
}

// MARK: Exposed functions

Tree copy(const Tree& tree)
{
    return Tree {
        .root = copy(tree.root),
        .type = tree.type,
        .stage = tree.stage,
        .requiresConversionData = tree.requiresConversionData
    };
}

} // namespace CSSCalc
} // namespace WebCore

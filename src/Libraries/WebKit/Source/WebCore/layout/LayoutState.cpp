/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 5, 2022.
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
#include "LayoutState.h"

#include "BlockFormattingState.h"
#include "InlineContentCache.h"
#include "LayoutBox.h"
#include "LayoutBoxGeometry.h"
#include "LayoutContainingBlockChainIterator.h"
#include "LayoutElementBox.h"
#include "LayoutInitialContainingBlock.h"
#include "RenderBox.h"
#include "TableFormattingState.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
namespace Layout {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(LayoutState);

LayoutState::LayoutState(const Document& document, const ElementBox& rootContainer, Type type, FormattingContextLayoutFunction&& formattingContextLayoutFunction, FormattingContextLogicalWidthFunction&& formattingContextLogicalWidthFunction , FormattingContextLogicalHeightFunction&& formattingContextLogicalHeightFunction)
    : m_type(type)
    , m_rootContainer(rootContainer)
    , m_securityOrigin(document.securityOrigin())
    , m_formattingContextLayoutFunction(WTFMove(formattingContextLayoutFunction))
    , m_formattingContextLogicalWidthFunction(WTFMove(formattingContextLogicalWidthFunction))
    , m_formattingContextLogicalHeightFunction(WTFMove(formattingContextLogicalHeightFunction))
{
    // It makes absolutely no sense to construct a dedicated layout state for a non-formatting context root (layout would be a no-op).
    ASSERT(root().establishesFormattingContext());

    updateQuirksMode(document);
}

LayoutState::~LayoutState() = default;

void LayoutState::updateQuirksMode(const Document& document)
{
    auto quirksMode = [&] {
        if (document.inLimitedQuirksMode())
            return LayoutState::QuirksMode::Limited;
        if (document.inQuirksMode())
            return LayoutState::QuirksMode::Yes;
        return LayoutState::QuirksMode::No;
    };
    setQuirksMode(quirksMode());
}

BoxGeometry& LayoutState::geometryForRootBox()
{
    return ensureGeometryForBox(root());
}

BoxGeometry& LayoutState::ensureGeometryForBoxSlow(const Box& layoutBox)
{
    if (LIKELY(m_type == Type::Primary)) {
#if ASSERT_ENABLED
        ASSERT(!layoutBox.m_cachedGeometryForPrimaryLayoutState);
        ASSERT(!layoutBox.m_primaryLayoutState);
        layoutBox.m_primaryLayoutState = this;
#endif
        layoutBox.m_cachedGeometryForPrimaryLayoutState = makeUnique<BoxGeometry>();
        return *layoutBox.m_cachedGeometryForPrimaryLayoutState;
    }

    return *m_layoutBoxToBoxGeometry.ensure(&layoutBox, [] {
        return makeUnique<BoxGeometry>();
    }).iterator->value;
}

bool LayoutState::hasFormattingState(const ElementBox& formattingContextRoot) const
{
    ASSERT(formattingContextRoot.establishesFormattingContext());
    return m_blockFormattingStates.contains(&formattingContextRoot) || m_tableFormattingStates.contains(&formattingContextRoot);
}

FormattingState& LayoutState::formattingStateForFormattingContext(const ElementBox& formattingContextRoot) const
{
    ASSERT(formattingContextRoot.establishesFormattingContext());

    if (formattingContextRoot.establishesBlockFormattingContext())
        return formattingStateForBlockFormattingContext(formattingContextRoot);

    if (formattingContextRoot.establishesTableFormattingContext())
        return formattingStateForTableFormattingContext(formattingContextRoot);

    CRASH();
}

BlockFormattingState& LayoutState::formattingStateForBlockFormattingContext(const ElementBox& blockFormattingContextRoot) const
{
    ASSERT(blockFormattingContextRoot.establishesBlockFormattingContext());
    return *m_blockFormattingStates.get(&blockFormattingContextRoot);
}

TableFormattingState& LayoutState::formattingStateForTableFormattingContext(const ElementBox& tableFormattingContextRoot) const
{
    ASSERT(tableFormattingContextRoot.establishesTableFormattingContext());
    return *m_tableFormattingStates.get(&tableFormattingContextRoot);
}

InlineContentCache& LayoutState::inlineContentCache(const ElementBox& formattingContextRoot)
{
    ASSERT(formattingContextRoot.establishesInlineFormattingContext());
    return *m_inlineContentCaches.ensure(&formattingContextRoot, [&] { return makeUnique<InlineContentCache>(); }).iterator->value;
}

BlockFormattingState& LayoutState::ensureBlockFormattingState(const ElementBox& formattingContextRoot)
{
    ASSERT(formattingContextRoot.establishesBlockFormattingContext());
    return *m_blockFormattingStates.ensure(&formattingContextRoot, [&] { return makeUnique<BlockFormattingState>(*this, formattingContextRoot); }).iterator->value;
}

TableFormattingState& LayoutState::ensureTableFormattingState(const ElementBox& formattingContextRoot)
{
    ASSERT(formattingContextRoot.establishesTableFormattingContext());
    return *m_tableFormattingStates.ensure(&formattingContextRoot, [&] { return makeUnique<TableFormattingState>(*this, formattingContextRoot); }).iterator->value;
}

void LayoutState::destroyBlockFormattingState(const ElementBox& formattingContextRoot)
{
    ASSERT(formattingContextRoot.establishesBlockFormattingContext());
    m_blockFormattingStates.remove(&formattingContextRoot);
}

void LayoutState::destroyInlineContentCache(const ElementBox& formattingContextRoot)
{
    ASSERT(formattingContextRoot.establishesInlineFormattingContext());
    m_inlineContentCaches.remove(&formattingContextRoot);
}

void LayoutState::layoutWithFormattingContextForBox(const ElementBox& box, std::optional<LayoutUnit> widthConstraint) const
{
    const_cast<LayoutState&>(*this).m_formattingContextLayoutFunction(box, widthConstraint, const_cast<LayoutState&>(*this));
}

LayoutUnit LayoutState::logicalWidthWithFormattingContextForBox(const ElementBox& box, LayoutIntegration::LogicalWidthType logicalWidthType) const
{
    return const_cast<LayoutState&>(*this).m_formattingContextLogicalWidthFunction(box, logicalWidthType);
}

LayoutUnit LayoutState::logicalHeightWithFormattingContextForBox(const ElementBox& box, LayoutIntegration::LogicalHeightType logicalHeightType) const
{
    return const_cast<LayoutState&>(*this).m_formattingContextLogicalHeightFunction(box, logicalHeightType);
}

}
}


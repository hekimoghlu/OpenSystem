/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 6, 2025.
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
#include "RenderTreeBuilderList.h"

#include "LegacyInlineIterator.h"
#include "LineInlineHeaders.h"
#include "RenderChildIterator.h"
#include "RenderListMarker.h"
#include "RenderMenuList.h"
#include "RenderMultiColumnFlow.h"
#include "RenderTable.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RenderTreeBuilder::List);

// FIXME: This shouldn't need LegacyInlineIterator
static bool generatesLineBoxesForInlineChild(RenderBlock& current, RenderObject* inlineObj)
{
    LegacyInlineIterator it(&current, inlineObj, 0);
    while (!it.atEnd() && !requiresLineBox(it))
        it.increment();
    return !it.atEnd();
}

static RenderBlock* getParentOfFirstLineBox(RenderBlock& current, RenderObject& marker)
{
    bool inQuirksMode = current.document().inQuirksMode();
    for (auto& child : childrenOfType<RenderObject>(current)) {
        if (&child == &marker)
            continue;

        if (child.isInline() && (!is<RenderInline>(child) || generatesLineBoxesForInlineChild(current, &child)))
            return &current;

        if (child.isFloating() || child.isOutOfFlowPositioned() || is<RenderMenuList>(child))
            continue;

        if (!is<RenderBlock>(child) || is<RenderTable>(child))
            break;

        if (auto* renderBox = dynamicDowncast<RenderBox>(child); renderBox && renderBox->isWritingModeRoot())
            break;

        if (is<RenderListItem>(current) && inQuirksMode && child.node() && isHTMLListElement(*child.node()))
            break;

        if (RenderBlock* lineBox = getParentOfFirstLineBox(downcast<RenderBlock>(child), marker))
            return lineBox;
    }

    return nullptr;
}

static RenderObject* firstNonMarkerChild(RenderBlock& parent)
{
    RenderObject* child = parent.firstChild();
    while (is<RenderListMarker>(child))
        child = child->nextSibling();
    return child;
}

RenderTreeBuilder::List::List(RenderTreeBuilder& builder)
    : m_builder(builder)
{
}

void RenderTreeBuilder::List::updateItemMarker(RenderListItem& listItemRenderer)
{
    auto& style = listItemRenderer.style();

    if (style.listStyleType().type == ListStyleType::Type::None && (!style.listStyleImage() || style.listStyleImage()->errorOccurred())) {
        if (auto* marker = listItemRenderer.markerRenderer())
            m_builder.destroy(*marker);
        return;
    }

    auto newStyle = listItemRenderer.computeMarkerStyle();
    RenderPtr<RenderListMarker> newMarkerRenderer;
    auto* markerRenderer = listItemRenderer.markerRenderer();
    if (markerRenderer)
        markerRenderer->setStyle(WTFMove(newStyle));
    else {
        newMarkerRenderer = WebCore::createRenderer<RenderListMarker>(listItemRenderer, WTFMove(newStyle));
        newMarkerRenderer->initializeStyle();
        markerRenderer = newMarkerRenderer.get();
        listItemRenderer.setMarkerRenderer(*markerRenderer);
    }

    RenderElement* currentParent = markerRenderer->parent();
    RenderBlock* newParent = getParentOfFirstLineBox(listItemRenderer, *markerRenderer);
    if (!newParent) {
        // If the marker is currently contained inside an anonymous box,
        // then we are the only item in that anonymous box (since no line box
        // parent was found). It's ok to just leave the marker where it is
        // in this case.
        if (currentParent && currentParent->isAnonymousBlock())
            return;
        if (auto* multiColumnFlow = listItemRenderer.multiColumnFlow())
            newParent = multiColumnFlow;
        else
            newParent = &listItemRenderer;
    }

    if (newParent == currentParent)
        return;

    if (currentParent)
        m_builder.attach(*newParent, m_builder.detach(*currentParent, *markerRenderer, WillBeDestroyed::No, RenderTreeBuilder::CanCollapseAnonymousBlock::No), firstNonMarkerChild(*newParent));
    else
        m_builder.attach(*newParent, WTFMove(newMarkerRenderer), firstNonMarkerChild(*newParent));

    // If current parent is an anonymous block that has lost all its children, destroy it.
    if (currentParent && currentParent->isAnonymousBlock() && !currentParent->firstChild() && !downcast<RenderBlock>(*currentParent).continuation())
        m_builder.destroy(*currentParent);
}

}

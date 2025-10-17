/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 20, 2023.
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
#include "LayoutElementBox.h"

#include "RenderElement.h"
#include "RenderStyleInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
namespace Layout {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ElementBox);

ElementBox::ElementBox(ElementAttributes&& attributes, RenderStyle&& style, std::unique_ptr<RenderStyle>&& firstLineStyle, OptionSet<BaseTypeFlag> baseTypeFlags)
    : Box(WTFMove(attributes), WTFMove(style), WTFMove(firstLineStyle), baseTypeFlags | ElementBoxFlag)
{
}

ElementBox::ElementBox(ElementAttributes&& attributes, OptionSet<ListMarkerAttribute> listMarkerAttributes, RenderStyle&& style, std::unique_ptr<RenderStyle>&& firstLineStyle)
    : Box(WTFMove(attributes), WTFMove(style), WTFMove(firstLineStyle), ElementBoxFlag)
    , m_replacedData(makeUnique<ReplacedData>())
{
    ASSERT(isListMarkerBox());
    m_replacedData->listMarkerAttributes = listMarkerAttributes;
}

ElementBox::ElementBox(ElementAttributes&& attributes, ReplacedAttributes&& replacedAttributes, RenderStyle&& style, std::unique_ptr<RenderStyle>&& firstLineStyle)
    : Box(WTFMove(attributes), WTFMove(style), WTFMove(firstLineStyle), ElementBoxFlag)
    , m_replacedData(makeUnique<ReplacedData>())
{
    m_replacedData->intrinsicSize = replacedAttributes.intrinsicSize;
    m_replacedData->intrinsicRatio = replacedAttributes.intrinsicRatio;
    m_replacedData->cachedImage = replacedAttributes.cachedImage;
}

ElementBox::~ElementBox()
{
    destroyChildren();
}

const Box* ElementBox::firstInFlowChild() const
{
    if (auto* firstChild = this->firstChild()) {
        if (firstChild->isInFlow())
            return firstChild;
        return firstChild->nextInFlowSibling();
    }
    return nullptr;
}

const Box* ElementBox::firstInFlowOrFloatingChild() const
{
    if (auto* firstChild = this->firstChild()) {
        if (firstChild->isInFlow() || firstChild->isFloatingPositioned())
            return firstChild;
        return firstChild->nextInFlowOrFloatingSibling();
    }
    return nullptr;
}

const Box* ElementBox::firstOutOfFlowChild() const
{
    if (auto* firstChild = this->firstChild()) {
        if (firstChild->isOutOfFlowPositioned())
            return firstChild;
        return firstChild->nextOutOfFlowSibling();
    }
    return nullptr;
}

const Box* ElementBox::lastInFlowChild() const
{
    if (auto* lastChild = this->lastChild()) {
        if (lastChild->isInFlow())
            return lastChild;
        return lastChild->previousInFlowSibling();
    }
    return nullptr;
}

const Box* ElementBox::lastInFlowOrFloatingChild() const
{
    if (auto* lastChild = this->lastChild()) {
        if (lastChild->isInFlow() || lastChild->isFloatingPositioned())
            return lastChild;
        return lastChild->previousInFlowOrFloatingSibling();
    }
    return nullptr;
}

const Box* ElementBox::lastOutOfFlowChild() const
{
    if (auto* lastChild = this->lastChild()) {
        if (lastChild->isOutOfFlowPositioned())
            return lastChild;
        return lastChild->previousOutOfFlowSibling();
    }
    return nullptr;
}

bool ElementBox::hasOutOfFlowChild() const
{
    return !!firstOutOfFlowChild();
}

void ElementBox::appendChild(UniqueRef<Box> childRef)
{
    insertChild(WTFMove(childRef), m_lastChild.get());
}

void ElementBox::insertChild(UniqueRef<Box> childRef, Box* beforeChild)
{
    auto childBox = childRef.moveToUniquePtr();
    ASSERT(!childBox->m_parent);
    ASSERT(!childBox->m_previousSibling);
    ASSERT(!childBox->m_nextSibling);

    childBox->m_parent = this;

    if (!m_firstChild || (beforeChild && !beforeChild->m_nextSibling)) {
        // Append as first and/or last.
        childBox->m_previousSibling = m_lastChild;
        auto& nextOrFirst = m_lastChild ? m_lastChild->m_nextSibling : m_firstChild;
        ASSERT(!nextOrFirst);

        m_lastChild = childBox.get();
        nextOrFirst = WTFMove(childBox);
        return;
    }

    if (!beforeChild) {
        // Insert as first.
        ASSERT(m_firstChild && m_lastChild);
        m_firstChild->m_previousSibling = childBox.get();
        childBox->m_nextSibling = WTFMove(m_firstChild);
        m_firstChild = WTFMove(childBox);
        return;
    }

    ASSERT(&beforeChild->parent() == this);
    auto* nextSibling = beforeChild->m_nextSibling.get();
    ASSERT(nextSibling);
    childBox->m_previousSibling = beforeChild;
    childBox->m_nextSibling = WTFMove(beforeChild->m_nextSibling);
    nextSibling->m_previousSibling = childBox.get();
    beforeChild->m_nextSibling = WTFMove(childBox);
}

void ElementBox::destroyChildren()
{
    m_lastChild = nullptr;

    auto childToDestroy = std::exchange(m_firstChild, nullptr);
    while (childToDestroy) {
        childToDestroy->m_parent = nullptr;
        childToDestroy->m_previousSibling = nullptr;
        if (childToDestroy->m_nextSibling)
            childToDestroy->m_nextSibling->m_previousSibling = nullptr;
        childToDestroy = std::exchange(childToDestroy->m_nextSibling, nullptr);
    }
}

bool ElementBox::hasIntrinsicWidth() const
{
    return (m_replacedData && m_replacedData->intrinsicSize) || style().logicalWidth().isIntrinsic();
}

bool ElementBox::hasIntrinsicHeight() const
{
    return (m_replacedData && m_replacedData->intrinsicSize) || style().logicalHeight().isIntrinsic();
}

bool ElementBox::hasIntrinsicRatio() const
{
    if (!hasAspectRatio())
        return false;
    return m_replacedData && (m_replacedData->intrinsicSize || m_replacedData->intrinsicRatio);
}

LayoutUnit ElementBox::intrinsicWidth() const
{
    ASSERT(hasIntrinsicWidth());
    if (m_replacedData && m_replacedData->intrinsicSize)
        return m_replacedData->intrinsicSize->width();
    return LayoutUnit { style().logicalWidth().value() };
}

LayoutUnit ElementBox::intrinsicHeight() const
{
    ASSERT(hasIntrinsicHeight());
    if (m_replacedData && m_replacedData->intrinsicSize)
        return m_replacedData->intrinsicSize->height();
    return LayoutUnit { style().logicalHeight().value() };
}

LayoutUnit ElementBox::intrinsicRatio() const
{
    ASSERT(hasIntrinsicRatio() || (hasIntrinsicWidth() && hasIntrinsicHeight()));
    if (m_replacedData) {
        if (m_replacedData->intrinsicRatio)
            return *m_replacedData->intrinsicRatio;
        if (m_replacedData->intrinsicSize->height())
            return m_replacedData->intrinsicSize->width() / m_replacedData->intrinsicSize->height();
    }
    return 1;
}

bool ElementBox::hasAspectRatio() const
{
    return isImage();
}

RenderElement* ElementBox::rendererForIntegration() const
{
    return downcast<RenderElement>(Box::rendererForIntegration());
}

}
}

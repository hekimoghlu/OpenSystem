/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 21, 2021.
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
#include "AccessibilityList.h"

#include "AXObjectCache.h"
#include "HTMLElement.h"
#include "HTMLNames.h"
#include "ListStyleType.h"
#include "PseudoElement.h"
#include "RenderListItem.h"
#include "RenderStyleInlines.h"

namespace WebCore {
    
using namespace HTMLNames;

AccessibilityList::AccessibilityList(AXID axID, RenderObject& renderer)
    : AccessibilityRenderObject(axID, renderer)
{
}

AccessibilityList::AccessibilityList(AXID axID, Node& node)
    : AccessibilityRenderObject(axID, node)
{
}

AccessibilityList::~AccessibilityList() = default;

Ref<AccessibilityList> AccessibilityList::create(AXID axID, RenderObject& renderer)
{
    return adoptRef(*new AccessibilityList(axID, renderer));
}

Ref<AccessibilityList> AccessibilityList::create(AXID axID, Node& node)
{
    return adoptRef(*new AccessibilityList(axID, node));
}

bool AccessibilityList::computeIsIgnored() const
{
    return isIgnoredByDefault();
}
    
bool AccessibilityList::isUnorderedList() const
{
    // The ARIA spec says the "list" role is supposed to mimic a UL or OL tag.
    // Since it can't be both, it's probably OK to say that it's an un-ordered list.
    // On the Mac, there's no distinction to the client.
    if (ariaRoleAttribute() == AccessibilityRole::List)
        return true;

    auto* node = this->node();
    return node && (node->hasTagName(menuTag) || node->hasTagName(ulTag));
}

bool AccessibilityList::isOrderedList() const
{
    // ARIA says a directory is like a static table of contents, which sounds like an ordered list.
    if (ariaRoleAttribute() == AccessibilityRole::Directory)
        return true;

    auto* node = this->node();
    return node && node->hasTagName(olTag);
}

bool AccessibilityList::isDescriptionList() const
{
    auto* node = this->node();
    return node && node->hasTagName(dlTag);
}

bool AccessibilityList::childHasPseudoVisibleListItemMarkers(Node* node)
{
    // Check if the list item has a pseudo-element that should be accessible (e.g. an image or text)
    auto* element = dynamicDowncast<Element>(node);
    RefPtr beforePseudo = element ? element->beforePseudoElement() : nullptr;
    if (!beforePseudo)
        return false;

    RefPtr axBeforePseudo = axObjectCache()->getOrCreate(beforePseudo->renderer());
    if (!axBeforePseudo)
        return false;
    
    if (!axBeforePseudo->isIgnored())
        return true;
    
    for (const auto& child : axBeforePseudo->unignoredChildren()) {
        if (!child->isIgnored())
            return true;
    }
    
    // Platforms which expose rendered text content through the parent element will treat
    // those renderers as "ignored" objects.
#if USE(ATSPI)
    String text = axBeforePseudo->textUnderElement();
    return !text.isEmpty() && !text.containsOnly<isASCIIWhitespace>();
#else
    return false;
#endif
}

AccessibilityRole AccessibilityList::determineAccessibilityRole()
{
    if (!m_childrenDirty && childrenInitialized())
        return determineAccessibilityRoleWithCleanChildren();

    m_ariaRole = determineAriaRoleAttribute();
    return isDescriptionList() ? AccessibilityRole::DescriptionList : AccessibilityRole::List;
}

AccessibilityRole AccessibilityList::determineAccessibilityRoleWithCleanChildren()
{
    ASSERT(!m_childrenDirty && childrenInitialized());
    m_ariaRole = determineAriaRoleAttribute();

    // Directory is mapped to list for now, but does not adhere to the same heuristics.
    if (ariaRoleAttribute() == AccessibilityRole::Directory)
        return AccessibilityRole::List;

    // Heuristic to determine if an ambiguous list is relevant to convey to the accessibility tree.
    //   1. If it's an ordered list or has role="list" defined, then it's a list.
    //      1a. Unless the list has no children, then it's not a list.
    //   2. If it is contained in <nav> or <el role="navigation">, it's a list.
    //   3. If it displays visible list markers, it's a list.
    //   4. If it does not display list markers, it's not a list.
    //   5. If it has one or zero listitem children, it's not a list.
    //   6. Otherwise it's a list.

    auto role = AccessibilityRole::List;

    // Temporarily set role so that we can query children (otherwise canHaveChildren returns false).
    SetForScope temporaryRole(m_role, role);

    unsigned listItemCount = 0;
    bool hasVisibleMarkers = false;

    const auto& children = unignoredChildren();
    // DescriptionLists are always semantically a description list, so do not apply heuristics.
    if (isDescriptionList() && children.size())
        return AccessibilityRole::DescriptionList;

    for (const auto& child : children) {
        RefPtr node = child->node();
        auto* axChild = dynamicDowncast<AccessibilityObject>(child.get());
        if (axChild && axChild->ariaRoleAttribute() == AccessibilityRole::ListItem)
            listItemCount++;
        else if (child->roleValue() == AccessibilityRole::ListItem) {
            // Rendered list items always count.
            if (auto* childRenderer = child->renderer(); childRenderer && childRenderer->isRenderListItem()) {
                if (!hasVisibleMarkers && (childRenderer->style().listStyleType().type != ListStyleType::Type::None || childRenderer->style().listStyleImage() || childHasPseudoVisibleListItemMarkers(childRenderer->node())))
                    hasVisibleMarkers = true;
                listItemCount++;
            } else if (node && node->hasTagName(liTag)) {
                // Inline elements that are in a list with an explicit role should also count.
                if (m_ariaRole == AccessibilityRole::List)
                    listItemCount++;

                if (childHasPseudoVisibleListItemMarkers(node.get())) {
                    hasVisibleMarkers = true;
                    listItemCount++;
                }
            }
        }
    }

    // Non <ul> lists and ARIA lists only need to have one child.
    // <ul>, <ol> lists need to have visible markers.
    if (ariaRoleAttribute() != AccessibilityRole::Unknown) {
        if (!listItemCount)
            role = AccessibilityRole::Group;
    } else if (!hasVisibleMarkers) {
        // http://webkit.org/b/193382 lists inside of navigation hierarchies should still be considered lists.
        if (Accessibility::findAncestor<AccessibilityObject>(*this, false, [] (auto& object) { return object.roleValue() == AccessibilityRole::LandmarkNavigation; }))
            role = AccessibilityRole::List;
        else
            role = AccessibilityRole::Group;
    }

    return role;
}

} // namespace WebCore

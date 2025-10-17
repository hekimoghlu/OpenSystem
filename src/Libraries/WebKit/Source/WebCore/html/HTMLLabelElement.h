/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 11, 2024.
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

#include "HTMLElement.h"

namespace WebCore {

class HTMLLabelElement final : public HTMLElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLLabelElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLLabelElement);
public:
    static Ref<HTMLLabelElement> create(const QualifiedName&, Document&);
    static Ref<HTMLLabelElement> create(Document&);

    WEBCORE_EXPORT RefPtr<HTMLElement> control() const;
    WEBCORE_EXPORT HTMLFormElement* form() const;

    bool willRespondToMouseClickEventsWithEditability(Editability) const final;
    void updateLabel(TreeScope&, const AtomString& oldForAttributeValue, const AtomString& newForAttributeValue);

private:
    HTMLLabelElement(const QualifiedName&, Document&);

    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode& parentOfInsertedTree) final;
    void removedFromAncestor(RemovalType, ContainerNode& oldParentOfRemovedTree) final;

    bool isEventTargetedAtInteractiveDescendants(Event&) const;

    bool accessKeyAction(bool sendMouseEvents) final;

    // Overridden to update the hover/active state of the corresponding control.
    void setActive(bool, Style::InvalidationScope) final;
    void setHovered(bool, Style::InvalidationScope, HitTestRequest) final;

    // Overridden to either click() or focus() the corresponding control.
    void defaultEventHandler(Event&) final;

    void focus(const FocusOptions&) final;

    bool isInteractiveContent() const final { return true; }

    bool m_processingClick { false };
};

} //namespace

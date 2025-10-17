/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 5, 2025.
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

class HTMLSlotElement;
class ToggleEventTask;

class HTMLDetailsElement final : public HTMLElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLDetailsElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLDetailsElement);
public:
    static Ref<HTMLDetailsElement> create(const QualifiedName& tagName, Document&);
    ~HTMLDetailsElement();

    void toggleOpen() { setBooleanAttribute(HTMLNames::openAttr, !hasAttribute(HTMLNames::openAttr)); }

    bool isActiveSummary(const HTMLSummaryElement&) const;

    void queueDetailsToggleEventTask(ToggleState oldState, ToggleState newState);

private:
    HTMLDetailsElement(const QualifiedName&, Document&);

    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) final;
    void didFinishInsertingNode() final;

    Vector<RefPtr<HTMLDetailsElement>> otherElementsInNameGroup();
    void ensureDetailsExclusivityAfterMutation();
    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;

    void didAddUserAgentShadowRoot(ShadowRoot&) final;
    bool isInteractiveContent() const final { return true; }

    WeakPtr<HTMLSlotElement, WeakPtrImplWithEventTargetData> m_summarySlot;
    WeakPtr<HTMLSummaryElement, WeakPtrImplWithEventTargetData> m_defaultSummary;
    RefPtr<HTMLSlotElement> m_defaultSlot;

    RefPtr<ToggleEventTask> m_toggleEventTask;
};

} // namespace WebCore

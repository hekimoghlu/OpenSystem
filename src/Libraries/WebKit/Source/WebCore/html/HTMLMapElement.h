/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 3, 2021.
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

class HitTestResult;
class HTMLImageElement;
    
class HTMLMapElement final : public HTMLElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLMapElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLMapElement);
public:
    static Ref<HTMLMapElement> create(Document&);
    static Ref<HTMLMapElement> create(const QualifiedName&, Document&);
    virtual ~HTMLMapElement();

    const AtomString& getName() const { return m_name; }

    bool mapMouseEvent(LayoutPoint location, const LayoutSize&, HitTestResult&);
    
    RefPtr<HTMLImageElement> imageElement();
    WEBCORE_EXPORT Ref<HTMLCollection> areas();

private:
    HTMLMapElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;

    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) final;
    void removedFromAncestor(RemovalType, ContainerNode&) final;

    AtomString m_name;
};

} // namespaces

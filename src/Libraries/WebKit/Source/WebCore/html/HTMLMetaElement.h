/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 24, 2022.
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

#include "Color.h"
#include "HTMLElement.h"
#include "MediaQuery.h"

namespace WebCore {

class HTMLMetaElement final : public HTMLElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLMetaElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLMetaElement);
public:
    static Ref<HTMLMetaElement> create(Document&);
    static Ref<HTMLMetaElement> create(const QualifiedName&, Document&);

    const AtomString& content() const;
    const AtomString& httpEquiv() const;
    const AtomString& name() const;

    bool mediaAttributeMatches();

    const Color& contentColor();

private:
    HTMLMetaElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason = AttributeModificationReason::Directly) final;
    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) final;
    void didFinishInsertingNode();
    void removedFromAncestor(RemovalType, ContainerNode&) final;

    void process(const AtomString& oldValue = nullAtom());

    std::optional<MQ::MediaQueryList> m_mediaQueryList;

    std::optional<Color> m_contentColor;
};

} // namespace WebCore

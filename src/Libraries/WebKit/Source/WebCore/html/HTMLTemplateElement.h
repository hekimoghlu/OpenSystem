/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 15, 2022.
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

class DocumentFragment;
class TemplateContentDocumentFragment;

class HTMLTemplateElement final : public HTMLElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLTemplateElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLTemplateElement);
public:
    static Ref<HTMLTemplateElement> create(const QualifiedName&, Document&);
    virtual ~HTMLTemplateElement();

    DocumentFragment& fragmentForInsertion() const;
    DocumentFragment& content() const;
    DocumentFragment* contentIfAvailable() const;

    const AtomString& shadowRootMode() const;
    void setShadowRootMode(const AtomString&);

    void setDeclarativeShadowRoot(ShadowRoot&);
    void attachAsDeclarativeShadowRootIfNeeded(Element&);

private:
    HTMLTemplateElement(const QualifiedName&, Document&);

    Ref<Node> cloneNodeInternal(TreeScope&, CloningOperation) final;
    void didMoveToNewDocument(Document& oldDocument, Document& newDocument) final;

    mutable RefPtr<TemplateContentDocumentFragment> m_content;
    WeakPtr<ShadowRoot, WeakPtrImplWithEventTargetData> m_declarativeShadowRoot;
};

} // namespace WebCore

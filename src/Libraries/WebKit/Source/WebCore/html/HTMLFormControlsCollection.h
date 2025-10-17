/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 23, 2023.
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

#include "CachedHTMLCollection.h"
#include "HTMLFormElement.h"
#include "RadioNodeList.h"

namespace WebCore {

class FormListedElement;
class HTMLImageElement;

// This class is just a big hack to find form elements even in malformed HTML elements.
// The famous <table><tr><form><td> problem.

class HTMLFormControlsCollection final : public CachedHTMLCollection<HTMLFormControlsCollection, CollectionTypeTraits<CollectionType::FormControls>::traversalType> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLFormControlsCollection);
public:
    static Ref<HTMLFormControlsCollection> create(ContainerNode&, CollectionType);
    virtual ~HTMLFormControlsCollection();

    HTMLElement* item(unsigned offset) const override;
    std::optional<std::variant<RefPtr<RadioNodeList>, RefPtr<Element>>> namedItemOrItems(const AtomString&) const;

    HTMLFormElement& ownerNode() const;

    // For CachedHTMLCollection.
    HTMLElement* customElementAfter(Element*) const;

private:
    explicit HTMLFormControlsCollection(ContainerNode&);

    void invalidateCacheForDocument(Document&) override;
    void updateNamedElementCache() const override;

    mutable CheckedPtr<Element> m_cachedElement;
    mutable unsigned m_cachedElementOffsetInArray { 0 };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_HTMLCOLLECTION(HTMLFormControlsCollection, CollectionType::FormControls)

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 27, 2022.
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
#include "HTMLOptionElement.h"
#include "HTMLSelectElement.h"

namespace WebCore {

class HTMLOptionsCollection final : public CachedHTMLCollection<HTMLOptionsCollection, CollectionTypeTraits<CollectionType::SelectOptions>::traversalType> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(HTMLOptionsCollection, WEBCORE_EXPORT);
public:
    using Base = CachedHTMLCollection<HTMLOptionsCollection, CollectionTypeTraits<CollectionType::SelectOptions>::traversalType>;

    static Ref<HTMLOptionsCollection> create(HTMLSelectElement&, CollectionType);

    inline HTMLSelectElement& selectElement();
    inline const HTMLSelectElement& selectElement() const;

    WEBCORE_EXPORT unsigned length() const final;
    WEBCORE_EXPORT HTMLOptionElement* item(unsigned offset) const final;
    WEBCORE_EXPORT HTMLOptionElement* namedItem(const AtomString& name) const final;
    
    ExceptionOr<void> setItem(unsigned index, HTMLOptionElement*);
    
    using OptionOrOptGroupElement = std::variant<RefPtr<HTMLOptionElement>, RefPtr<HTMLOptGroupElement>>;
    using HTMLElementOrInt = std::variant<RefPtr<HTMLElement>, int>;
    WEBCORE_EXPORT ExceptionOr<void> add(const OptionOrOptGroupElement&, const std::optional<HTMLElementOrInt>& before);
    WEBCORE_EXPORT void remove(int index);

    WEBCORE_EXPORT int selectedIndex() const;
    WEBCORE_EXPORT void setSelectedIndex(int);

    WEBCORE_EXPORT ExceptionOr<void> setLength(unsigned);

    // For CachedHTMLCollection.
    inline bool elementMatches(Element&) const;

private:
    explicit HTMLOptionsCollection(HTMLSelectElement&);
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_HTMLCOLLECTION(HTMLOptionsCollection, CollectionType::SelectOptions)

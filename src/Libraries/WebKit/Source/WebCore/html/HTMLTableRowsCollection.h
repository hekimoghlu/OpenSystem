/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 17, 2023.
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
#include "HTMLTableElement.h"

namespace WebCore {

class HTMLTableRowElement;

class HTMLTableRowsCollection final : public CachedHTMLCollection<HTMLTableRowsCollection, CollectionTypeTraits<CollectionType::TableRows>::traversalType> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLTableRowsCollection);
public:
    static Ref<HTMLTableRowsCollection> create(HTMLTableElement&, CollectionType);

    HTMLTableElement& tableElement() { return downcast<HTMLTableElement>(ownerNode()); }
    const HTMLTableElement& tableElement() const { return downcast<HTMLTableElement>(ownerNode()); }

    static HTMLTableRowElement* rowAfter(HTMLTableElement&, HTMLTableRowElement*);
    static HTMLTableRowElement* lastRow(HTMLTableElement&);

    // For CachedHTMLCollection.
    Element* customElementAfter(Element*) const;

private:
    explicit HTMLTableRowsCollection(HTMLTableElement&);
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_HTMLCOLLECTION(HTMLTableRowsCollection, CollectionType::TableRows)

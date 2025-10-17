/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 14, 2022.
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

namespace WebCore {

template <CollectionTraversalType traversalType>
class GenericCachedHTMLCollection final : public CachedHTMLCollection<GenericCachedHTMLCollection<traversalType>, traversalType> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_TEMPLATE(GenericCachedHTMLCollection);
    static_assert(traversalType != CollectionTraversalType::CustomForwardOnly, "CustomForwardOnly should use non GenericCachedHTMLCollection.");
public:
    static Ref<GenericCachedHTMLCollection> create(ContainerNode& base, CollectionType collectionType)
    {
        return adoptRef(*new GenericCachedHTMLCollection(base, collectionType));
    }

    virtual ~GenericCachedHTMLCollection();

    bool elementMatches(Element&) const;

private:
    GenericCachedHTMLCollection(ContainerNode&, CollectionType);
};

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_TEMPLATE_IMPL(template<CollectionTraversalType traversalType>, GenericCachedHTMLCollection<traversalType>);

} // namespace WebCore

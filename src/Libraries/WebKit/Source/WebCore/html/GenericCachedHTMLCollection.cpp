/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 1, 2023.
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
#include "GenericCachedHTMLCollection.h"

#include "CachedHTMLCollectionInlines.h"
#include "HTMLFieldSetElement.h"
#include "HTMLNames.h"
#include "HTMLObjectElement.h"
#include "HTMLOptionElement.h"

namespace WebCore {

using namespace HTMLNames;

template <CollectionTraversalType traversalType>
GenericCachedHTMLCollection<traversalType>::GenericCachedHTMLCollection(ContainerNode& base, CollectionType collectionType)
    : CachedHTMLCollection<GenericCachedHTMLCollection<traversalType>, traversalType>(base, collectionType)
{ }
template GenericCachedHTMLCollection<CollectionTraversalType::Descendants>::GenericCachedHTMLCollection(ContainerNode&, CollectionType);
template GenericCachedHTMLCollection<CollectionTraversalType::ChildrenOnly>::GenericCachedHTMLCollection(ContainerNode&, CollectionType);

template <CollectionTraversalType traversalType>
GenericCachedHTMLCollection<traversalType>::~GenericCachedHTMLCollection() = default;
template GenericCachedHTMLCollection<CollectionTraversalType::Descendants>::~GenericCachedHTMLCollection();
template GenericCachedHTMLCollection<CollectionTraversalType::ChildrenOnly>::~GenericCachedHTMLCollection();

template <CollectionTraversalType traversalType>
bool GenericCachedHTMLCollection<traversalType>::elementMatches(Element& element) const
{
    switch (this->type()) {
    case CollectionType::NodeChildren:
        return true;
    case CollectionType::DocImages:
        return element.hasTagName(imgTag);
    case CollectionType::DocScripts:
        return element.hasTagName(scriptTag);
    case CollectionType::DocForms:
        return element.hasTagName(formTag);
    case CollectionType::TableTBodies:
        return element.hasTagName(tbodyTag);
    case CollectionType::TRCells:
        return element.hasTagName(tdTag) || element.hasTagName(thTag);
    case CollectionType::TSectionRows:
        return element.hasTagName(trTag);
    case CollectionType::SelectedOptions: {
        auto* optionElement = dynamicDowncast<HTMLOptionElement>(element);
        return optionElement && optionElement->selected();
    }
    case CollectionType::DataListOptions:
        return is<HTMLOptionElement>(element);
    case CollectionType::MapAreas:
        return element.hasTagName(areaTag);
    case CollectionType::DocEmbeds:
        return element.hasTagName(embedTag);
    case CollectionType::DocLinks:
        return (element.hasTagName(aTag) || element.hasTagName(areaTag)) && element.hasAttributeWithoutSynchronization(hrefAttr);
    case CollectionType::DocAnchors:
        return element.hasTagName(aTag) && element.hasAttributeWithoutSynchronization(nameAttr);
    case CollectionType::FieldSetElements:
        return element.isFormListedElement();
    case CollectionType::ByClass:
    case CollectionType::ByTag:
    case CollectionType::ByHTMLTag:
    case CollectionType::AllDescendants:
    case CollectionType::DocAll:
    case CollectionType::DocEmpty:
    case CollectionType::DocumentAllNamedItems:
    case CollectionType::DocumentNamedItems:
    case CollectionType::FormControls:
    case CollectionType::SelectOptions:
    case CollectionType::TableRows:
    case CollectionType::WindowNamedItems:
        break;
    }
    // Remaining collection types have their own CachedHTMLCollection subclasses and are not using GenericCachedHTMLCollection.
    ASSERT_NOT_REACHED();
    return false;
}

template bool GenericCachedHTMLCollection<CollectionTraversalType::Descendants>::elementMatches(Element&) const;
template bool GenericCachedHTMLCollection<CollectionTraversalType::ChildrenOnly>::elementMatches(Element&) const;

} // namespace WebCore

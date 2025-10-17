/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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
#include "HTMLOptionsCollection.h"

#include "CachedHTMLCollectionInlines.h"
#include "HTMLOptionsCollectionInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLOptionsCollection);

HTMLOptionsCollection::HTMLOptionsCollection(HTMLSelectElement& select)
    : CachedHTMLCollection(select, CollectionType::SelectOptions)
{
}

Ref<HTMLOptionsCollection> HTMLOptionsCollection::create(HTMLSelectElement& select, CollectionType)
{
    return adoptRef(*new HTMLOptionsCollection(select));
}

ExceptionOr<void> HTMLOptionsCollection::add(const OptionOrOptGroupElement& element, const std::optional<HTMLElementOrInt>& before)
{
    return selectElement().add(element, before);
}

void HTMLOptionsCollection::remove(int index)
{
    selectElement().remove(index);
}

int HTMLOptionsCollection::selectedIndex() const
{
    return selectElement().selectedIndex();
}

void HTMLOptionsCollection::setSelectedIndex(int index)
{
    selectElement().setSelectedIndex(index);
}

unsigned HTMLOptionsCollection::length() const
{
    return Base::length();
}

ExceptionOr<void> HTMLOptionsCollection::setLength(unsigned length)
{
    return selectElement().setLength(length);
}

HTMLOptionElement* HTMLOptionsCollection::item(unsigned offset) const
{
    return downcast<HTMLOptionElement>(Base::item(offset));
}

HTMLOptionElement* HTMLOptionsCollection::namedItem(const AtomString& name) const
{
    return downcast<HTMLOptionElement>(Base::namedItem(name));
}

ExceptionOr<void> HTMLOptionsCollection::setItem(unsigned index, HTMLOptionElement* optionElement)
{
    return selectElement().setItem(index, optionElement);
}

} //namespace

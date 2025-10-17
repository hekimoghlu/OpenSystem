/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 7, 2023.
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

#include "SVGList.h"

namespace WebCore {

template<typename PropertyType>
class SVGPropertyList : public SVGList<Ref<PropertyType>>, public SVGPropertyOwner {
public:
    using BaseList = SVGList<Ref<PropertyType>>;
    using BaseList::isEmpty;
    using BaseList::size;
    using BaseList::append;

protected:
    using SVGPropertyOwner::SVGPropertyOwner;
    using BaseList::m_items;
    using BaseList::m_access;
    using BaseList::m_owner;

    SVGPropertyList(SVGPropertyOwner* owner = nullptr, SVGPropertyAccess access = SVGPropertyAccess::ReadWrite)
        : BaseList(owner, access)
    {
    }

    ~SVGPropertyList()
    {
        // Detach the items from the list before it is deleted.
        detachItems();
    }

    void detachItems() override
    {
        for (auto& item : m_items)
            item->detach();
    }

    SVGPropertyOwner* owner() const override { return m_owner; }

    void commitPropertyChange(SVGProperty*) override
    {
        if (owner())
            owner()->commitPropertyChange(this);
    }

    Ref<PropertyType> at(unsigned index) const override
    {
        ASSERT(index < size());
        return m_items.at(index).copyRef();
    }

    Ref<PropertyType> insert(unsigned index, Ref<PropertyType>&& newItem) override
    {
        ASSERT(index <= size());

        // Spec: if newItem is not a detached object, then set newItem to be
        // a clone object of newItem.
        if (newItem->isAttached())
            newItem = newItem->clone();

        // Spec: Attach newItem to the list object.
        newItem->attach(this, m_access);
        m_items.insert(index, WTFMove(newItem));
        return at(index);
    }

    Ref<PropertyType> replace(unsigned index, Ref<PropertyType>&& newItem) override
    {
        ASSERT(index < size());
        Ref<PropertyType>& item = m_items[index];

        // Spec: Detach item.
        item->detach();

        // Spec: if newItem is not a detached object, then set newItem to be
        // a clone object of newItem.
        if (newItem->isAttached())
            item = newItem->clone();
        else
            item = WTFMove(newItem);

        // Spec: Attach newItem to the list object.
        item->attach(this, m_access);
        return at(index);
    }

    Ref<PropertyType> remove(unsigned index) override
    {
        ASSERT(index < size());
        Ref<PropertyType> item = at(index);

        // Spec: Detach item.
        item->detach();
        m_items.remove(index);
        return item;
    }

    Ref<PropertyType> append(Ref<PropertyType>&& newItem) override
    {
        // Spec: if newItem is not a detached object, then set newItem to be
        // a clone object of newItem.
        if (newItem->isAttached())
            newItem = newItem->clone();

        // Spec: Attach newItem to the list object.
        newItem->attach(this, m_access);
        m_items.append(WTFMove(newItem));
        return at(size() - 1);
    }
};

}

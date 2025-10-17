/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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

#include "ExceptionOr.h"
#include "SVGProperty.h"

namespace WebCore {

template<typename ItemType>
class SVGList : public SVGProperty {
public:
    unsigned length() const { return numberOfItems(); }

    unsigned numberOfItems() const
    {
        return m_items.size();
    }

    ExceptionOr<void> clear()
    {
        auto result = canAlterList();
        if (result.hasException())
            return result.releaseException();
        ASSERT(result.releaseReturnValue());

        clearItems();
        commitChange();
        return { };
    }

    ExceptionOr<ItemType> getItem(unsigned index)
    {
        auto result = canGetItem(index);
        if (result.hasException())
            return result.releaseException();
        ASSERT(result.releaseReturnValue());

        return at(index);
    }

    ExceptionOr<ItemType> initialize(ItemType&& newItem)
    {
        auto result = canAlterList();
        if (result.hasException())
            return result.releaseException();

        // Spec: Clears all existing current items from the list.
        clearItems();

        auto item = append(WTFMove(newItem));
        commitChange();
        return item;
    }

    ExceptionOr<ItemType> insertItemBefore(ItemType&& newItem, unsigned index)
    {
        auto result = canAlterList();
        if (result.hasException())
            return result.releaseException();
        ASSERT(result.releaseReturnValue());

        // Spec: If the index is greater than or equal to numberOfItems,
        // then the new item is appended to the end of the list.
        if (index > numberOfItems())
            index = numberOfItems();

        auto item = insert(index, WTFMove(newItem));
        commitChange();
        return item;
    }

    ExceptionOr<ItemType> replaceItem(ItemType&& newItem, unsigned index)
    {
        auto result = canReplaceItem(index);
        if (result.hasException())
            return result.releaseException();
        ASSERT(result.releaseReturnValue());

        auto item = replace(index, WTFMove(newItem));
        commitChange();
        return item;
    }

    ExceptionOr<ItemType> removeItem(unsigned index)
    {
        auto result = canRemoveItem(index);
        if (result.hasException())
            return result.releaseException();
        ASSERT(result.releaseReturnValue());

        auto item = remove(index);
        commitChange();
        return item;
    }

    ExceptionOr<ItemType> appendItem(ItemType&& newItem)
    {
        auto result = canAlterList();
        if (result.hasException())
            return result.releaseException();
        ASSERT(result.releaseReturnValue());

        auto item = append(WTFMove(newItem));
        commitChange();
        return item;
    }

    ExceptionOr<void> setItem(unsigned index, ItemType&& newItem)
    {
        auto result = replaceItem(WTFMove(newItem), index);
        if (result.hasException())
            return result.releaseException();
        return { };
    }

    bool isSupportedPropertyIndex(unsigned index) const { return index < m_items.size(); }

    // Parsers and animators need to have a direct access to the items.
    Vector<ItemType>& items() { return m_items; }
    const Vector<ItemType>& items() const { return m_items; }
    size_t size() const { return m_items.size(); }
    bool isEmpty() const { return m_items.isEmpty(); }

    void clearItems()
    {
        detachItems();
        m_items.clear();
    }

protected:
    using SVGProperty::SVGProperty;

    ExceptionOr<bool> canAlterList() const
    {
        if (isReadOnly())
            return Exception { ExceptionCode::NoModificationAllowedError };
        return true;
    }

    ExceptionOr<bool> canGetItem(unsigned index)
    {
        if (index >= m_items.size())
            return Exception { ExceptionCode::IndexSizeError };
        return true;
    }

    ExceptionOr<bool> canReplaceItem(unsigned index)
    {
        auto result = canAlterList();
        if (result.hasException())
            return result.releaseException();
        ASSERT(result.releaseReturnValue());

        if (index >= m_items.size())
            return Exception { ExceptionCode::IndexSizeError };
        return true;
    }

    ExceptionOr<bool> canRemoveItem(unsigned index)
    {
        auto result = canAlterList();
        if (result.hasException())
            return result.releaseException();
        ASSERT(result.releaseReturnValue());

        if (index >= m_items.size())
            return Exception { ExceptionCode::IndexSizeError };
        return true;
    }

    virtual void detachItems() { }
    virtual ItemType at(unsigned index) const = 0;
    virtual ItemType insert(unsigned index, ItemType&&) = 0;
    virtual ItemType replace(unsigned index, ItemType&&) = 0;
    virtual ItemType remove(unsigned index) = 0;
    virtual ItemType append(ItemType&&) = 0;

    Vector<ItemType> m_items;
};

}

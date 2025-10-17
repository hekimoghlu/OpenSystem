/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 29, 2021.
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
class SVGPrimitiveList : public SVGList<PropertyType> {
protected:
    using Base = SVGList<PropertyType>;
    using Base::Base;
    using Base::size;
    using Base::m_items;

    PropertyType at(unsigned index) const override
    {
        ASSERT(index < size());
        return m_items.at(index);
    }

    PropertyType insert(unsigned index, PropertyType&& newItem) override
    {
        ASSERT(index <= size());
        m_items.insert(index, WTFMove(newItem));
        return at(index);
    }

    PropertyType replace(unsigned index, PropertyType&& newItem) override
    {
        ASSERT(index < size());
        m_items.at(index) = WTFMove(newItem);
        return at(index);
    }

    PropertyType remove(unsigned index) override
    {
        ASSERT(index < size());
        PropertyType item = at(index);
        m_items.remove(index);
        return item;
    }

    PropertyType append(PropertyType&& newItem) override
    {
        m_items.append(WTFMove(newItem));
        return at(size() - 1);
    }
};

}

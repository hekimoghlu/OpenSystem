/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 19, 2022.
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

#include "SVGPropertyList.h"

namespace WebCore {

template<typename PropertyType>
class SVGValuePropertyList : public SVGPropertyList<PropertyType> {
public:
    using Base = SVGPropertyList<PropertyType>;
    using Base::clearItems;
    using Base::items;
    using Base::size;

    SVGValuePropertyList& operator=(const SVGValuePropertyList& other)
    {
        clearItems();
        for (const auto& item : other.items())
            append(PropertyType::create(item->value()));
        return *this;
    }

    // This casting operator returns a Vector of the underlying ValueType,
    // for example Vector<float> from SVGNumberList.
    operator Vector<typename PropertyType::ValueType>() const
    {
        Vector<typename PropertyType::ValueType> values;
        for (const auto& item : items())
            values.append(item->value());
        return values;
    }

    void resize(size_t newSize)
    {
        // Add new items.
        while (size() < newSize)
            append(PropertyType::create());

        // Remove existing items.
        while (size() > newSize)
            remove(size() - 1);
    }

protected:
    using Base::append;
    using Base::remove;

    // Base and default constructor. Do not use "using Base::Base" because of Windows and GTK ports.
    SVGValuePropertyList(SVGPropertyOwner* owner = nullptr, SVGPropertyAccess access = SVGPropertyAccess::ReadWrite)
        : Base(owner, access)
    {
    }

    // Used by SVGAnimatedPropertyList when creating it animVal from baseVal.
    SVGValuePropertyList(const SVGValuePropertyList& other, SVGPropertyAccess access = SVGPropertyAccess::ReadWrite)
        : Base(other.owner(), access)
    {
        // Clone all items.
        for (const auto& item : other.items())
            append(PropertyType::create(item->value()));
    }
};

}

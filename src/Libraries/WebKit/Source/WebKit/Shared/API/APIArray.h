/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 4, 2024.
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

#include "APIObject.h"
#include <wtf/Forward.h>
#include <wtf/IteratorAdaptors.h>
#include <wtf/IteratorRange.h>
#include <wtf/Vector.h>

namespace API {

class Array final : public ObjectImpl<Object::Type::Array> {
private:
    template <class T>
    struct IsTypePredicate {
        bool operator()(const RefPtr<Object>& object) const { return object->type() == T::APIType; }
    };

    template <class T>
    struct GetObjectTransform {
        const T* operator()(const RefPtr<Object>& object) const { return static_cast<const T*>(object.get()); }
    };

    template <typename T>
    using ElementsOfTypeRange = WTF::IteratorRange<WTF::TransformIterator<GetObjectTransform<T>, WTF::FilterIterator<IsTypePredicate<T>, Vector<RefPtr<Object>>::const_iterator>>>;

public:
    static Ref<Array> create();
    static Ref<Array> createWithCapacity(size_t);
    static Ref<Array> create(Vector<RefPtr<Object>>&&);
    static Ref<Array> createStringArray(const Vector<WTF::String>&);
    static Ref<Array> createStringArray(const std::span<const WTF::String>);
    Vector<WTF::String> toStringVector();
    Ref<Array> copy();

    template<typename T>
    T* at(size_t i) const
    {
        if (!m_elements[i] || m_elements[i]->type() != T::APIType)
            return nullptr;

        return static_cast<T*>(m_elements[i].get());
    }

    Object* at(size_t i) const { return m_elements[i].get(); }
    size_t size() const { return m_elements.size(); }

    const Vector<RefPtr<Object>>& elements() const { return m_elements; }
    Vector<RefPtr<Object>>& elements() { return m_elements; }

    template<typename T>
    ElementsOfTypeRange<T> elementsOfType() const
    {
        return WTF::makeIteratorRange(
            WTF::makeTransformIterator(GetObjectTransform<T>(), WTF::makeFilterIterator(IsTypePredicate<T>(), m_elements.begin(), m_elements.end())),
            WTF::makeTransformIterator(GetObjectTransform<T>(), WTF::makeFilterIterator(IsTypePredicate<T>(), m_elements.end(), m_elements.end()))
        );
    }

    template<typename MatchFunction>
    unsigned removeAllMatching(const MatchFunction& matchFunction)
    {
        return m_elements.removeAllMatching(matchFunction);
    }

    template<typename T, typename MatchFunction>
    unsigned removeAllOfTypeMatching(NOESCAPE const MatchFunction& matchFunction)
    {
        return m_elements.removeAllMatching([&] (const RefPtr<Object>& object) -> bool {
            if (object->type() != T::APIType)
                return false;
            return matchFunction(static_pointer_cast<T>(object));
        });
    }

private:
    explicit Array(Vector<RefPtr<Object>>&& elements)
        : m_elements(WTFMove(elements))
    {
    }

    Vector<RefPtr<Object>> m_elements;
};

} // namespace API

SPECIALIZE_TYPE_TRAITS_API_OBJECT(Array);

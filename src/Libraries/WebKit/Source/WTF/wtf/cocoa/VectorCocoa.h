/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 5, 2022.
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

#include <wtf/BlockPtr.h>
#include <wtf/Forward.h>
#include <wtf/Vector.h>
#include <wtf/cocoa/SpanCocoa.h>

namespace WTF {

// Specialize the behavior of these functions by overloading the makeNSArrayElement
// functions and makeVectorElement functions. The makeNSArrayElement function takes
// a const& to a collection element and can return either a RetainPtr<id> or an id
// if the value is autoreleased. The makeVectorElement function takes an ignored
// pointer to the vector element type, making argument-dependent lookup work, and an
// id for the array element, and returns an std::optional<T> of the the vector element,
// allowing us to filter out array elements that are not of the expected type.
//
//    RetainPtr<id> makeNSArrayElement(const CollectionElementType& collectionElement);
//        -or-
//    id makeNSArrayElement(const VectorElementType& vectorElement);
//
//    std::optional<VectorElementType> makeVectorElement(const VectorElementType*, id arrayElement);

template<typename CollectionType> RetainPtr<NSMutableArray> createNSArray(CollectionType&&);
template<typename VectorElementType> Vector<VectorElementType> makeVector(NSArray *);

// This overload of createNSArray takes a function to map each vector element to an Objective-C object.
// The map function has the same interface as the makeNSArrayElement function above, but can be any
// function including a lambda, a function-like object, or Function<>.
template<typename CollectionType, typename MapFunctionType> RetainPtr<NSMutableArray> createNSArray(CollectionType&&, NOESCAPE MapFunctionType&&);

// This overload of makeVector takes a function to map each Objective-C object to a vector element.
// Currently, the map function needs to return an Optional.
template<typename MapFunctionType> Vector<typename std::invoke_result_t<MapFunctionType, id>::value_type> makeVector(NSArray *, NOESCAPE MapFunctionType&&);

// Implementation details of the function templates above.

inline void addUnlessNil(NSMutableArray *array, id value)
{
    if (value)
        [array addObject:value];
}

template<typename CollectionType> RetainPtr<NSMutableArray> createNSArray(CollectionType&& collection)
{
    auto array = adoptNS([[NSMutableArray alloc] initWithCapacity:std::size(collection)]);
    for (auto&& element : collection)
        addUnlessNil(array.get(), getPtr(makeNSArrayElement(std::forward<decltype(element)>(element))));
    return array;
}

template<typename CollectionType, typename MapFunctionType> RetainPtr<NSMutableArray> createNSArray(CollectionType&& collection, NOESCAPE MapFunctionType&& function)
{
    auto array = adoptNS([[NSMutableArray alloc] initWithCapacity:std::size(collection)]);
    for (auto&& element : std::forward<CollectionType>(collection)) {
        if constexpr (std::is_rvalue_reference_v<CollectionType&&> && !std::is_const_v<std::remove_reference_t<decltype(element)>>)
            addUnlessNil(array.get(), getPtr(function(WTFMove(element))));
        else
            addUnlessNil(array.get(), getPtr(function(element)));
    }
    return array;
}

template<typename VectorElementType> Vector<VectorElementType> makeVector(NSArray *array)
{
    if (!array)
        return { };

    Vector<VectorElementType> vector;
    vector.reserveInitialCapacity(array.count);
    for (id element in array) {
        constexpr const VectorElementType* typedNull = nullptr;
        if (auto vectorElement = makeVectorElement(typedNull, element))
            vector.append(WTFMove(*vectorElement));
    }
    vector.shrinkToFit();
    return vector;
}

template<typename MapFunctionType> Vector<typename std::invoke_result_t<MapFunctionType, id>::value_type> makeVector(NSArray *array, NOESCAPE MapFunctionType&& function)
{
    if (!array)
        return { };

    Vector<typename std::invoke_result_t<MapFunctionType, id>::value_type> vector;
    vector.reserveInitialCapacity(array.count);
    for (id element in array) {
        if (auto vectorElement = std::invoke(std::forward<MapFunctionType>(function), element))
            vector.append(WTFMove(*vectorElement));
    }
    vector.shrinkToFit();
    return vector;
}

inline Vector<uint8_t> makeVector(NSData *data)
{
    return span(data);
}

template<typename T>
inline RetainPtr<dispatch_data_t> makeDispatchData(Vector<T>&& vector)
{
    auto buffer = vector.releaseBuffer();
    auto span = buffer.span();
    return adoptNS(dispatch_data_create(span.data(), span.size_bytes(), dispatch_get_main_queue(), makeBlockPtr([buffer = WTFMove(buffer)] { }).get()));
}

} // namespace WTF

using WTF::createNSArray;
using WTF::makeDispatchData;
using WTF::makeVector;

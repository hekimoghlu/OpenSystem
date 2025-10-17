/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 8, 2022.
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

#include <wtf/HashTraits.h>
#include <wtf/OptionSet.h>

namespace WTF {

template<typename T> struct DefaultHash<OptionSet<T>> {
    static unsigned hash(OptionSet<T> key)
    {
        return IntHash<typename OptionSet<T>::StorageType>::hash(key.toRaw());
    }

    static bool equal(OptionSet<T> a, OptionSet<T> b)
    {
        return a == b;
    }

    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

template<typename T> struct HashTraits<OptionSet<T>> : GenericHashTraits<OptionSet<T>> {
    using StorageTraits = UnsignedWithZeroKeyHashTraits<typename OptionSet<T>::StorageType>;

    static OptionSet<T> emptyValue()
    {
        return OptionSet<T>::fromRaw(StorageTraits::emptyValue());
    }

    static void constructDeletedValue(OptionSet<T>& slot)
    {
        typename OptionSet<T>::StorageType storage;
        StorageTraits::constructDeletedValue(storage);
        slot = OptionSet<T>::fromRaw(storage);
    }

    static bool isDeletedValue(OptionSet<T> value)
    {
        return StorageTraits::isDeletedValue(value.toRaw());
    }
};

} // namespace WTF

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 22, 2024.
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

#include <CoreFoundation/CoreFoundation.h>
#include <wtf/Assertions.h>

#ifndef CF_BRIDGED_TYPE
#define CF_BRIDGED_TYPE(T)
#endif

namespace WTF {

template <typename> struct CFTypeTrait;

// Use dynamic_cf_cast<> instead of checked_cf_cast<> when actively checking CF types,
// similar to dynamic_cast<> in C++. Be sure to include a nullptr check.

template<typename T> T dynamic_cf_cast(CFTypeRef object)
{
    if (!object)
        return nullptr;

    if (CFGetTypeID(object) != CFTypeTrait<T>::typeID())
        return nullptr;

    return static_cast<T>(const_cast<CF_BRIDGED_TYPE(id) void*>(object));
}

template<typename T, typename U> RetainPtr<T> dynamic_cf_cast(RetainPtr<U>&& object)
{
    if (!object)
        return nullptr;

    if (CFGetTypeID(object.get()) != CFTypeTrait<T>::typeID())
        return nullptr;

    return adoptCF(static_cast<T>(const_cast<CF_BRIDGED_TYPE(id) void*>(object.leakRef())));
}

// Use checked_cf_cast<> instead of dynamic_cf_cast<> when a specific CF type is required.

template<typename T> T checked_cf_cast(CFTypeRef object)
{
    if (!object)
        return nullptr;

    RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(CFGetTypeID(object) == CFTypeTrait<T>::typeID());

    return static_cast<T>(const_cast<CF_BRIDGED_TYPE(id) void*>(object));
}

} // namespace WTF

#define WTF_DECLARE_CF_TYPE_TRAIT(ClassName) \
template <> \
struct WTF::CFTypeTrait<ClassName##Ref> { \
    static inline CFTypeID typeID(void) { return ClassName##GetTypeID(); } \
};

WTF_DECLARE_CF_TYPE_TRAIT(CFArray);
WTF_DECLARE_CF_TYPE_TRAIT(CFBoolean);
WTF_DECLARE_CF_TYPE_TRAIT(CFData);
WTF_DECLARE_CF_TYPE_TRAIT(CFDictionary);
WTF_DECLARE_CF_TYPE_TRAIT(CFNumber);
WTF_DECLARE_CF_TYPE_TRAIT(CFString);
WTF_DECLARE_CF_TYPE_TRAIT(CFURL);

#define WTF_DECLARE_CF_MUTABLE_TYPE_TRAIT(ClassName, MutableClassName) \
template <> \
struct WTF::CFTypeTrait<MutableClassName##Ref> { \
    static inline CFTypeID typeID(void) { return ClassName##GetTypeID(); } \
};

WTF_DECLARE_CF_MUTABLE_TYPE_TRAIT(CFArray, CFMutableArray);
WTF_DECLARE_CF_MUTABLE_TYPE_TRAIT(CFData, CFMutableData);
WTF_DECLARE_CF_MUTABLE_TYPE_TRAIT(CFDictionary, CFMutableDictionary);
WTF_DECLARE_CF_MUTABLE_TYPE_TRAIT(CFString, CFMutableString);

#undef WTF_DECLARE_CF_MUTABLE_TYPE_TRAIT

using WTF::checked_cf_cast;
using WTF::dynamic_cf_cast;

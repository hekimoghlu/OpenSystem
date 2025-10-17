/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 25, 2022.
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

#include <memory>
#include <wtf/Box.h>

namespace WTF {

template<typename T> struct BoxPtrDeleter {
    void operator()(T*) const = delete;
};

#define WTF_DEFINE_BOXPTR_DELETER(typeName, deleterFunction) \
    template<> struct BoxPtrDeleter<typeName> \
    { \
        void operator() (typeName* ptr) const \
        { \
            deleterFunction(ptr); \
        } \
    };

template<typename T> using BoxPtr = Box<std::unique_ptr<T, BoxPtrDeleter<T>>>;

template<typename T> BoxPtr<T> adoptInBoxPtr(T* ptr)
{
    return BoxPtr<T>::create(ptr);
}

template<typename T> bool operator==(const BoxPtr<T>& lhs, const BoxPtr<T>& rhs)
{
    if (!lhs && !rhs)
        return true;

    if (!lhs || !rhs)
        return false;

    if (!lhs->get() && !rhs->get())
        return true;

    if (!lhs->get() || !rhs->get())
        return false;

    return *lhs == *rhs;
}

} // namespace WTF

using WTF::adoptInBoxPtr;
using WTF::BoxPtr;

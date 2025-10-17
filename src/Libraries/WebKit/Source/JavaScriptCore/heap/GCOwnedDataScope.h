/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 1, 2022.
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

#include "EnsureStillAliveHere.h"

namespace JSC {

class JSCell;

// This class is used return data owned by a JSCell. Consider:
// int foo(JSString* jsString)
// {
//     String& string = jsString->value(globalObject);
//     // Do something that triggers a GC (e.g. any object allocation)
//     return string.length();
// }
// The C++ compiler is technically free to "drop" the last reference to jsString
// after we call value. Since our GC relies on live C++ values still existing on
// the stack when we trigger a GC we could collect jsString freeing string and
// causing a UAF. This class helps avoid that problem by forcing the C++ compiler
// to keep a reference to the owner GCed object of our data while being as
// unobtrusive as possible. That said there are a few caveats for callers:
//   1) **NEVER** do:
//          Data& data = foo->getData();
//      or:
//          Data* data = foo->getData();
//      this will **NOT** prevent the GC from collecting foo before data goes
//      out of scope.
//
//   2) The preferred way to call a function returning a GCOwnedDataScope is:
//          auto data = foo->getData();
//      this will ensure foo exists long enough prevent a GC before data goes
//      out of scope.
//
//
//   3) It's ok to bind to / return a retained value:
//          Data data = foo->getData();
//      or:
//          return foo->getData();
//      These aren't ideal as they trigger a retain and maybe release on data
//      but this can't be avoided in some cases.

template<typename T>
struct GCOwnedDataScope {
    GCOwnedDataScope(const JSCell* cell, T value)
        : owner(cell)
        , data(value)
    { }

    ~GCOwnedDataScope() { ensureStillAliveHere(owner); }

    operator const T() const { return data; }
    operator T() { return data; }

    std::remove_reference_t<T>* operator->() requires (!std::is_pointer_v<T>) { return &data; }
    const std::remove_reference_t<T>* operator->() const requires (!std::is_pointer_v<T>) { return &data; }

    T operator->() requires (std::is_pointer_v<T>) { return data; }
    const T operator->() const requires (std::is_pointer_v<T>) { return data; }

    auto operator[](unsigned index) const { return data[index]; }

    // Convenience conversion for String -> StringView
    operator StringView() const requires (std::is_same_v<std::decay_t<T>, String>) { return data; }

    const JSCell* owner;
    T data;
};

}

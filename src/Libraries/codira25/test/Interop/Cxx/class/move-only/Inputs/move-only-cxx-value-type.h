/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 26, 2024.
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

#ifndef TEST_INTEROP_CXX_CLASS_MOVE_ONLY_VT_H
#define TEST_INTEROP_CXX_CLASS_MOVE_ONLY_VT_H

#include <memory>

struct Copyable {
    int x;
};

struct NonCopyable {
    inline NonCopyable(int x) : x(x) {}
    inline NonCopyable(const NonCopyable &) = delete;
    inline NonCopyable(NonCopyable &&other) : x(other.x) { other.x = -123; }

    inline int method(int y) const { return x * y; }
    inline int mutMethod(int y) {
      x = y;
      return y;
    }

    int x;
};

struct NonCopyableDerived: public NonCopyable {
    NonCopyableDerived(int x) : NonCopyable(x) {}
};

struct NonCopyableDerivedDerived: public NonCopyableDerived {
    NonCopyableDerivedDerived(int x) : NonCopyableDerived(x) {}
};

struct NonCopyableHolder {
    inline NonCopyableHolder(int x) : x(x) {}
    inline NonCopyableHolder(const NonCopyableHolder &) = delete;
    inline NonCopyableHolder(NonCopyableHolder &&other) : x(std::move(other.x)) {}

    inline NonCopyable &returnMutNonCopyableRef() { return x; }

    inline const NonCopyable &returnNonCopyableRef() const { return x; }

    NonCopyable x;
};

struct NonCopyableHolderDerived: NonCopyableHolder {
    inline NonCopyableHolderDerived(int x) : NonCopyableHolder(x) {}
};

struct NonCopyableHolderDerivedDerived: NonCopyableHolderDerived {
    inline NonCopyableHolderDerivedDerived(int x) : NonCopyableHolderDerived(x) {}

    inline int getActualX() const {
        return x.x;
    }
};

inline NonCopyable *getNonCopyablePtr() { return nullptr; }
inline NonCopyableDerived *getNonCopyableDerivedPtr() { return nullptr; }

#endif // TEST_INTEROP_CXX_CLASS_MOVE_ONLY_VT_H

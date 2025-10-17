/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 26, 2024.
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

#ifndef TEST_INTEROP_CXX_OPERATORS_MOVE_ONLY_OPS_H
#define TEST_INTEROP_CXX_OPERATORS_MOVE_ONLY_OPS_H

#include <memory>

struct Copyable {
    int x;
};

struct NonCopyable {
    inline NonCopyable(int x) : x(x) {}
    inline NonCopyable(const NonCopyable &) = delete;
    inline NonCopyable(NonCopyable &&other) : x(other.x) { other.x = 0; }

    inline int method(int y) const { return x * y; }
    inline int mutMethod(int y) {
      x = y;
      return y;
    }

    int x;
};

#define NONCOPYABLE_HOLDER_WRAPPER(Name) \
private: \
NonCopyable x; \
public: \
inline Name(int x) : x(x) {} \
inline Name(const Name &) = delete; \
inline Name(Name &&other) : x(std::move(other.x)) {}

class NonCopyableHolderConstDeref {
    NONCOPYABLE_HOLDER_WRAPPER(NonCopyableHolderConstDeref)

    inline const NonCopyable & operator *() const { return x; }
};

class NonCopyableHolderPairedDeref {
    NONCOPYABLE_HOLDER_WRAPPER(NonCopyableHolderPairedDeref)

    inline const NonCopyable & operator *() const { return x; }
    inline NonCopyable & operator *() { return x; }
};

class NonCopyableHolderMutDeref {
    NONCOPYABLE_HOLDER_WRAPPER(NonCopyableHolderMutDeref)

    inline NonCopyable & operator *() { return x; }
};

class NonCopyableHolderValueConstDeref {
    NONCOPYABLE_HOLDER_WRAPPER(NonCopyableHolderValueConstDeref)

    inline NonCopyable operator *() const { return NonCopyable(x.x); }
};

class NonCopyableHolderValueMutDeref {
    NONCOPYABLE_HOLDER_WRAPPER(NonCopyableHolderValueMutDeref)

    inline NonCopyable operator *() { return NonCopyable(x.x); }
};

template<class T>
class OneDerived: public T {
public:
    OneDerived(int x) : T(x) {}
};

using NonCopyableHolderConstDerefDerivedDerived = OneDerived<OneDerived<NonCopyableHolderConstDeref>>;

using NonCopyableHolderPairedDerefDerivedDerived = OneDerived<OneDerived<NonCopyableHolderPairedDeref>>;

using NonCopyableHolderMutDerefDerivedDerived = OneDerived<OneDerived<NonCopyableHolderMutDeref>>;

using NonCopyableHolderValueConstDerefDerivedDerived = OneDerived<OneDerived<NonCopyableHolderValueConstDeref>>;

using NonCopyableHolderValueMutDerefDerivedDerived = OneDerived<OneDerived<NonCopyableHolderValueMutDeref>>;

#endif // TEST_INTEROP_CXX_OPERATORS_MOVE_ONLY_OPS_H

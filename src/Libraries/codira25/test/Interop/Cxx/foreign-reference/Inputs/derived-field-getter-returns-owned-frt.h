/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 1, 2022.
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

#define FRT                                       \
  __attribute__((language_attr("import_reference"))) \
  __attribute__((language_attr("retain:immortal")))  \
  __attribute__((language_attr("release:immortal")))

int &getLiveRefCountedCounter() {
    static int counter = 0;
    return counter;
}

class RefCounted {
public:
    RefCounted() { getLiveRefCountedCounter()++; }
    ~RefCounted() {
        getLiveRefCountedCounter()--;
    }

    void retain() {
        ++refCount;
    }
    void release() {
        --refCount;
        if (refCount == 0)
            delete this;
    }

    int testVal = 1;
private:
    int refCount = 1;
}   __attribute__((language_attr("import_reference")))
__attribute__((language_attr("retain:retainRefCounted")))
__attribute__((language_attr("release:releaseRefCounted")));

RefCounted * _Nonnull createRefCounted() {
    return new RefCounted;
}

void retainRefCounted(RefCounted *r) {
    if (r)
        r->retain();
}
void releaseRefCounted(RefCounted *r) {
    if (r)
        r->release();
}

class BaseFieldFRT {
public:
    BaseFieldFRT(): value(new RefCounted) {}
    BaseFieldFRT(const BaseFieldFRT &other): value(other.value) {
        value->retain();
    }
    ~BaseFieldFRT() {
        value->release();
    }

    RefCounted * _Nonnull value;
};

class DerivedFieldFRT : public BaseFieldFRT {
};

class NonEmptyBase {
public:
    int getY() const {
        return y;
    }
private:
    int y = 11;
};

class DerivedDerivedFieldFRT : public NonEmptyBase, public DerivedFieldFRT {
};

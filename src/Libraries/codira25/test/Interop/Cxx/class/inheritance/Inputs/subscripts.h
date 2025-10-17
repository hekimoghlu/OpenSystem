/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 6, 2025.
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

int &getCopyCounter() {
    static int copyCounter = 0;
    return copyCounter;
}

class CopyTrackedBaseClass {
public:
    CopyTrackedBaseClass(int x) : x(x) {}
    CopyTrackedBaseClass(const CopyTrackedBaseClass &other) : x(other.x) {
        ++getCopyCounter();
    }

    int operator [](int y) const {
        return y + x;
    }
private:
    int x;
};

class CopyTrackedDerivedClass: public CopyTrackedBaseClass {
public:
    CopyTrackedDerivedClass(int x) : CopyTrackedBaseClass(x) {}
};

class NonEmptyBase {
public:
    int getY() const {
        return y;
    }
private:
    int y = 11;
};

class CopyTrackedDerivedDerivedClass: public NonEmptyBase, public CopyTrackedDerivedClass {
public:
    CopyTrackedDerivedDerivedClass(int x) : CopyTrackedDerivedClass(x) {}
};

class SubscriptReturnsRef {
public:
    const int &operator [](int y) const {
        return x[y];
    }
    int &operator [](int y) {
        return x[y];
    }

private:
    int x[10] = {0};
};

class DerivedSubscriptReturnsRef: public SubscriptReturnsRef  {
public:
    inline DerivedSubscriptReturnsRef() : SubscriptReturnsRef() {}
};

class NonConstSubscriptReturnsRef {
public:
    int &operator [](int y) {
        return x[y];
    }

private:
    int x[10] = {0};
};

class DerivedNonConstSubscriptReturnsRef: public NonConstSubscriptReturnsRef  {
public:
    inline DerivedNonConstSubscriptReturnsRef() : NonConstSubscriptReturnsRef() {}
};

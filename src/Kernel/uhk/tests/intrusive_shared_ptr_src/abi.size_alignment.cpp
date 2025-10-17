/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 14, 2025.
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

//
// This tests that the alignment and size of a class are the same whether
// they have a `T*` or a shared pointer data member.
//

#include <libkern/c++/intrusive_shared_ptr.h>
#include "test_policy.h"
#include <cstddef>
#include <darwintest.h>


namespace ns1 {
struct FooShared {
	test_shared_ptr<int> ptr;
};

struct FooRaw {
	int* ptr;
};

static_assert(sizeof(FooShared) == sizeof(FooRaw));
static_assert(alignof(FooShared) == alignof(FooRaw));
static_assert(offsetof(FooShared, ptr) == offsetof(FooRaw, ptr));
}

namespace ns2 {
struct FooShared {
	int i;
	test_shared_ptr<int> ptr;
};

struct FooRaw {
	int i;
	int* ptr;
};

static_assert(sizeof(FooShared) == sizeof(FooRaw));
static_assert(alignof(FooShared) == alignof(FooRaw));
static_assert(offsetof(FooShared, ptr) == offsetof(FooRaw, ptr));
}

namespace ns3 {
struct FooShared {
	char c;
	test_shared_ptr<int> ptr;
	int i;
};

struct FooRaw {
	char c;
	int* ptr;
	int i;
};

static_assert(sizeof(FooShared) == sizeof(FooRaw));
static_assert(alignof(FooShared) == alignof(FooRaw));
static_assert(offsetof(FooShared, ptr) == offsetof(FooRaw, ptr));
}

namespace ns4 {
struct FooShared {
	char c;
	unsigned int b : 5;
	test_shared_ptr<int> ptr;
	int i;
};

struct FooRaw {
	char c;
	unsigned int b : 5;
	int* ptr;
	int i;
};

static_assert(sizeof(FooShared) == sizeof(FooRaw));
static_assert(alignof(FooShared) == alignof(FooRaw));
static_assert(offsetof(FooShared, ptr) == offsetof(FooRaw, ptr));
}

namespace ns5 {
struct __attribute__((packed)) FooShared {
	char c;
	unsigned int b : 5;
	test_shared_ptr<int> ptr;
	int i;
};

struct __attribute__((packed)) FooRaw {
	char c;
	unsigned int b : 5;
	int* ptr;
	int i;
};

static_assert(sizeof(FooShared) == sizeof(FooRaw));
static_assert(alignof(FooShared) == alignof(FooRaw));
static_assert(offsetof(FooShared, ptr) == offsetof(FooRaw, ptr));
}

namespace ns6 {
struct FooShared {
	char c;
	unsigned int b : 5;
	test_shared_ptr<int> ptr;
	int i __attribute__((packed));
};

struct FooRaw {
	char c;
	unsigned int b : 5;
	int* ptr;
	int i __attribute__((packed));
};

static_assert(sizeof(FooShared) == sizeof(FooRaw));
static_assert(alignof(FooShared) == alignof(FooRaw));
static_assert(offsetof(FooShared, ptr) == offsetof(FooRaw, ptr));
}

namespace ns7 {
struct FooShared {
	char c;
	unsigned int b : 5;
	test_shared_ptr<int> ptr __attribute__((packed));
	int i;
};

struct FooRaw {
	char c;
	unsigned int b : 5;
	int* ptr __attribute__((packed));
	int i;
};

static_assert(sizeof(FooShared) == sizeof(FooRaw));
static_assert(alignof(FooShared) == alignof(FooRaw));
static_assert(offsetof(FooShared, ptr) == offsetof(FooRaw, ptr));
}

T_DECL(abi_size_alignment, "intrusive_shared_ptr.abi.size_alignment", T_META_TAG_VM_PREFERRED) {
	T_PASS("intrusive_shared_ptr.abi.size_alignment compile-time tests passed");
}

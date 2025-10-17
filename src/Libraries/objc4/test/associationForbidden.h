/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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

#include "testroot.i"

@interface Normal : TestRoot
@end
@implementation Normal
@end

@interface Forbidden : TestRoot
@end
@implementation Forbidden
@end

struct minimal_unrealized_class {
    void * __ptrauth_objc_isa_pointer isa;
    void * __ptrauth_objc_super_pointer superclass;
    void *cachePtr;
    uintptr_t maskAndOccupied;
    struct minimal_class_ro * __ptrauth_objc_class_ro ro;
};

struct minimal_class_ro {
    uint32_t flags;
};

extern struct minimal_unrealized_class OBJC_CLASS_$_Forbidden;

#define RO_FORBIDS_ASSOCIATED_OBJECTS (1<<10)

static void *key = &key;

static void test(void);

int main()
{
    struct minimal_unrealized_class *localForbidden = &OBJC_CLASS_$_Forbidden;
    ptrauth_strip(localForbidden->ro, ptrauth_key_process_independent_data)->flags |= RO_FORBIDS_ASSOCIATED_OBJECTS;
    test();
}

static inline void ShouldSucceed(id obj) {
    objc_setAssociatedObject(obj, key, obj, OBJC_ASSOCIATION_ASSIGN);
    id assoc = objc_getAssociatedObject(obj, key);
    fprintf(stderr, "Associated object is %p\n", assoc);
    testassert(obj == assoc);
}

static inline void ShouldFail(id obj) {
    objc_setAssociatedObject(obj, key, obj, OBJC_ASSOCIATION_ASSIGN);
    fail("should have crashed trying to set the associated object");
}

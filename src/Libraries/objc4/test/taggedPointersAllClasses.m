/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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

// TEST_CONFIG

#include "test.h"
#include "testroot.i"
#include <objc/runtime.h>
#include <objc/objc-internal.h>

#if OBJC_HAVE_TAGGED_POINTERS

@interface TagSuperclass: TestRoot

- (void)test;

@end

@implementation TagSuperclass

- (void)test {}

@end

int expectedTag;
uintptr_t expectedPayload;
uintptr_t sawPayload;
int sawTag;

void impl(void *self, SEL cmd) {
    (void)cmd;
    testassert(expectedTag == _objc_getTaggedPointerTag(self));
    testassert(expectedPayload == _objc_getTaggedPointerValue(self));
    sawPayload = _objc_getTaggedPointerValue(self);
    sawTag = _objc_getTaggedPointerTag(self);
}

int main()
{
    Class classes[OBJC_TAG_Last52BitPayload + 1] = {};
    
    for (int i = 0; i <= OBJC_TAG_Last52BitPayload; i++) {
        objc_tag_index_t tag = (objc_tag_index_t)i;
        if (i > OBJC_TAG_Last60BitPayload && i < OBJC_TAG_First52BitPayload)
            continue;
        if (_objc_getClassForTag(tag) != nil)
            continue;
        
        char *name;
        asprintf(&name, "Tag%d", i);
        classes[i] = objc_allocateClassPair([TagSuperclass class], name, 0);
        free(name);
        
        class_addMethod(classes[i], @selector(test), (IMP)impl, "v@@");
        
        objc_registerClassPair(classes[i]);
        _objc_registerTaggedPointerClass(tag, classes[i]);
    }
    
    for (int i = 0; i <= OBJC_TAG_Last52BitPayload; i++) {
        objc_tag_index_t tag = (objc_tag_index_t)i;
        if (classes[i] == nil)
            continue;
        
        for (int byte = 0; byte <= 0xff; byte++) {
            uintptr_t payload;
            memset(&payload, byte, sizeof(payload));
            
            if (i <= OBJC_TAG_Last60BitPayload)
                payload >>= _OBJC_TAG_PAYLOAD_RSHIFT;
            else
                payload >>= _OBJC_TAG_EXT_PAYLOAD_RSHIFT;

            expectedTag = i;
            expectedPayload = payload;
            id obj = (__bridge id)_objc_makeTaggedPointer(tag, payload);
            [obj test];
            testassert(sawPayload == payload);
            testassert(sawTag == i);
        }
    }
    
    succeed(__FILE__);
}

#else

int main()
{
    succeed(__FILE__);
}

#endif

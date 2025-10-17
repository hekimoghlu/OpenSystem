/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 29, 2024.
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

// objc.h redefines these calls into bridge casts.
// This test verifies that the function implementations are exported.
__BEGIN_DECLS
extern void *retainedObject(void *arg) __asm__("_objc_retainedObject");
extern void *unretainedObject(void *arg) __asm__("_objc_unretainedObject");
extern void *unretainedPointer(void *arg) __asm__("_objc_unretainedPointer");
__END_DECLS

int main()
{
    void *p = (void*)&main;
    testassert(p == retainedObject(p));
    testassert(p == unretainedObject(p));
    testassert(p == unretainedPointer(p));
    succeed(__FILE__);
}

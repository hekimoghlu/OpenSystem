/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 12, 2023.
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
#include "test.h"
#include <string.h>
#include <objc/objc-runtime.h>
#include <objc/objc-auto.h>
#include <objc/objc-internal.h>

int main()
{
    // Make sure @selector values are correctly fixed up
    testassert(@selector(foo) == sel_registerName("foo"));

    // sel_getName recognizes the zero SEL
    testassert(0 == strcmp("<null selector>", sel_getName(0)));

    // sel_lookUpByName returns NULL for NULL string
    testassert(NULL == sel_lookUpByName(NULL));

    // sel_lookUpByName returns NULL for unregistered and matches later registered selector
    {
        SEL sel;
        testassert(NULL == sel_lookUpByName("__testSelectorLookUp:"));
        testassert(NULL != (sel = sel_registerName("__testSelectorLookUp:")));
        testassert(sel  == sel_lookUpByName("__testSelectorLookUp:"));
    }

    // sel_lookUpByName matches @selector value
    testassert(@selector(foo2) == sel_lookUpByName("foo2"));

    succeed(__FILE__);
}

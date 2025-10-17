/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 23, 2021.
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
//  byrefaccess.m
//  test that byref access to locals is accurate
//  testObjects
//
//  Created by Blaine Garst on 5/13/08.
//  Copyright 2008 __MyCompanyName__. All rights reserved.
//
// TEST_CONFIG

#include <stdio.h>
#include "test.h"

void callVoidVoid(void (^closure)(void)) {
    closure();
}

int main() {
    __block int i = 10;
    
    callVoidVoid(^{ ++i; });
    
    if (i != 11) {
        fail("didn't update i");
        return 1;
    }

    succeed(__FILE__);
}

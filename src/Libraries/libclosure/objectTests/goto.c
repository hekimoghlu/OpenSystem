/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 1, 2025.
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
/*
 *  goto.c
 *  testObjects
 *
 *  Created by Blaine Garst on 10/17/08.
 *  Copyright 2008 __MyCompanyName__. All rights reserved.
 *
 */
 
// TEST_CONFIG 
// rdar://6289031

#include <stdio.h>
#include "test.h"

int main()
{
    __block int val = 0;
    
    ^{ val = 1; }();
    
    if (val == 0) {
        goto out_bad; // error: local byref variable val is in the scope of this goto
    }
    
    succeed(__FILE__);

 out_bad:
    fail("val not updated!");
}

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 28, 2021.
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
 *  orbars.c
 *  testObjects
 *
 *  Created by Blaine Garst on 9/17/08.
 *  Copyright 2008 __MyCompanyName__. All rights reserved.
 */

// rdar://6276695 error: before â€˜|â€™ token
// TEST_CONFIG RUN=0

/*
TEST_BUILD_OUTPUT
.*orbars.c:29:\d+: error: expected expression
END
*/

#include <stdio.h>
#include "test.h"

int main() {
    int i __unused = 10;
    void (^b)(void) __unused = ^(void){ | i | printf("hello world, %d\n", i); };
    fail("should not compile");
}

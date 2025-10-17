/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 16, 2023.
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
 *  sizeof.c
 *  testObjects
 *
 *  Created by Blaine Garst on 2/17/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

// TEST_CONFIG RUN=0

/*
TEST_BUILD_OUTPUT
.*sizeof.c: In function 'main':
.*sizeof.c:36: error: invalid type argument of 'unary \*'
OR
.*sizeof.c: In function '.*main.*':
.*sizeof.c:36: error: invalid application of 'sizeof' to a function type
OR
.*sizeof.c:36:(47|51): error: indirection requires pointer operand \('void \(\^\)\((void)?\)' invalid\)
END
 */

#include <stdio.h>
#include "test.h"

int main() {
    void (^aBlock)(void) = ^{ printf("hellow world\n"); };

    fail("the size of a block is %ld", sizeof(*aBlock));
}

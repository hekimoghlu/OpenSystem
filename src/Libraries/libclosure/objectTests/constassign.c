/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 17, 2023.
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
//  constassign.c
//
//  Created by Blaine Garst on 3/21/08.
//  Copyright 2008 __MyCompanyName__. All rights reserved.

// TEST_CONFIG RUN=0

/*
TEST_BUILD_OUTPUT
.*constassign.c:38:12: error: cannot assign to variable 'blockA' with const-qualified type 'void \(\^const\)\((void)?\)'
.*constassign.c:37:18: note: .*
.*constassign.c:39:10: error: cannot assign to variable 'fptr' with const-qualified type 'void \(\*const\)\((void)?\)'
.*constassign.c:36:18: note: .*
END
*/



// shouldn't be able to assign to a const pointer
// CONFIG error: assignment of read-only

#import <stdio.h>
#import "test.h"

void foo(void) { printf("I'm in foo\n"); }
void bar(void) { printf("I'm in bar\n"); }

int main() {
    void (*const fptr)(void) = foo;
    void (^const  blockA)(void) = ^ { printf("hello\n"); };
    blockA = ^ { printf("world\n"); } ;
    fptr = bar;
    fail("should not compile");
}

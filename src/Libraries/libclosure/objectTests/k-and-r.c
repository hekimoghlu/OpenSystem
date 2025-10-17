/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 24, 2023.
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
// TEST_CONFIG RUN=0


/*
TEST_BUILD_OUTPUT
.*k-and-r.c:26:11: error: incompatible block pointer types assigning to 'char \(\^\)\(\)' from 'char \(\^\)\(char\)'
OR
.*k-and-r.c:26:11: error: assigning to 'char \(\^\)\(\)' from incompatible type 'char \(\^\)\(char\)'
.*k-and-r.c:27:20: error: too many arguments to block call, expected 0, have 1
.*k-and-r.c:28:20: error: too many arguments to block call, expected 0, have 1
END
*/

#import <stdio.h>
#import <stdlib.h>
#import "test.h"

int main() {
    char (^rot13)();
    rot13 = ^(char c) { return (char)(((c - 'a' + 13) % 26) + 'a'); };
    char n = rot13('a');
    char c = rot13('p');
    if ( n != 'n' || c != 'c' ) {
        fail("rot13('a') returned %c, rot13('p') returns %c\n", n, c);
    }

    fail("should not compile");
}

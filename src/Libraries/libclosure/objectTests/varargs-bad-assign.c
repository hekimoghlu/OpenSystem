/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 24, 2024.
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
.*varargs-bad-assign.c: In function 'main':
.*varargs-bad-assign.c:43: error: incompatible block pointer types assigning 'int \(\^\)\(int,  int,  int\)', expected 'int \(\^\)\(int\)'
OR
.*varargs-bad-assign.c: In function '.*main.*':
.*varargs-bad-assign.c:43: error: cannot convert 'int \(\^\)\(int, int, int, \.\.\.\)' to 'int \(\^\)\(int, \.\.\.\)' in assignment
OR
.*varargs-bad-assign.c:31:10: error:( incompatible block pointer types)? assigning to 'int \(\^\)\(int, \.\.\.\)' from( incompatible type)? 'int \(\^\)\(int, int, int, \.\.\.\)'
END
*/

#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <stdarg.h>
#import "test.h"

int main () {
    int (^sumn)(int n, ...);
    int six = 0;
    
    sumn = ^(int a __unused, int b __unused, int n, ...){
        int result = 0;
        va_list numbers;
        int i;

        va_start(numbers, n);
        for (i = 0 ; i < n ; i++) {
            result += va_arg(numbers, int);
        }
        va_end(numbers);

        return result;
    };

    six = sumn(3, 1, 2, 3);

    if ( six != 6 ) {
        fail("Expected 6 but got %d", six);
    }
    
    succeed(__FILE__);
}

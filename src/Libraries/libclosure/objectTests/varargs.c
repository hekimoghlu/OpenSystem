/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 9, 2022.
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
//  -*- mode:C; c-basic-offset:4; tab-width:4; intent-tabs-mode:nil;  -*-
// TEST_CONFIG

#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <stdarg.h>
#import "test.h"

int main () {
    int (^sumn)(int n, ...) = ^(int n, ...){
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
    int six = sumn(3, 1, 2, 3);
    
    if ( six != 6 ) {
        fail("Expected 6 but got %d", six);
    }

    succeed(__FILE__);
}

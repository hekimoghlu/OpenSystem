/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 2, 2024.
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
 *  variadic.c
 *  testObjects
 *
 *  Created by Blaine Garst on 2/17/09.
 *  Copyright 2009 Apple. All rights reserved.
 *
 */

// PURPOSE Test that variadic arguments compile and work for Blocks
// TEST_CONFIG

#import <stdarg.h>
#import <stdio.h>
#import "test.h"

int main() {
    
    long (^addthem)(const char *, ...) = ^long (const char *format, ...){
        va_list argp;
        const char *p;
        int i;
        char c;
        double d;
        long result = 0;
        va_start(argp, format);
        //printf("starting...\n");
        for (p = format; *p; p++) switch (*p) {
            case 'i':
                i = va_arg(argp, int);
                //printf("i: %d\n", i);
                result += i;
                break;
            case 'd':
                d = va_arg(argp, double);
                //printf("d: %g\n", d);
                result += (int)d;
                break;
            case 'c':
                c = va_arg(argp, int);
                //printf("c: '%c'\n", c);
                result += c;
                break;
        }
        //printf("...done\n\n");
        return result;
    };
    long testresult = addthem("ii", 10, 20);
    if (testresult != 30) {
        fail("got wrong result: %ld", testresult);
    }
    testresult = addthem("idc", 30, 40.0, 'a');
    if (testresult != (70+'a')) {
        fail("got different wrong result: %ld", testresult);
    }

    succeed(__FILE__);
}



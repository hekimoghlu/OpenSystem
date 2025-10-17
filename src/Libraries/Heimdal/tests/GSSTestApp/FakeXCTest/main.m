/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 25, 2022.
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
//  FakeXCTest
//
//  Copyright (c) 2014 Apple. All rights reserved.
//

#import "FakeXCTest.h"
#import <getopt.h>

int main(int argc, char **argv)
{
    int result = 1;
    int ch;

    static struct option longopts[] = {
        { "limit",      required_argument,      NULL,           'l' },
        { NULL,         0,                      NULL,           0   }
    };

    optind = 0;
    while ((ch = getopt_long(argc, argv, "l:", longopts, NULL)) != -1) {
        switch (ch) {
            case 'l':
                [XCTest setLimit:optarg];
                break;
        }
    }


    @autoreleasepool {
        result = [XCTest runTests];
    }
    return result;
}


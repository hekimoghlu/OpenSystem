/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 23, 2023.
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
//  NSFileHandle+Formatting.m
//  sec
//
//

#include <stdarg.h>

#import <Foundation/Foundation.h>
#import "NSFileHandle+Formatting.h"


@implementation NSFileHandle (Formatting)

- (void) writeString: (NSString*) string {
    [self writeData:[string dataUsingEncoding:NSUTF8StringEncoding]];
}

- (void) writeFormat: (NSString*) format, ... {
    va_list args;
    va_start(args, format);

    NSString* formatted = [[NSString alloc] initWithFormat:format arguments:args];

    va_end(args);

    [self writeString: formatted];
// Remove with <rdar://problem/28925164> Enable ARC wherever possible in Security.framework
#if !__has_feature(objc_arc)
    [formatted release];
#endif
}

@end

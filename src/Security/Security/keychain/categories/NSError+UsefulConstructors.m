/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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
#import "NSError+UsefulConstructors.h"

@implementation NSError (UsefulConstructors)

+ (instancetype)errorWithDomain:(NSErrorDomain)domain code:(NSInteger)code description:(NSString*)description {
    return [NSError errorWithDomain:domain code:code description:description underlying:nil];
}

+ (instancetype)errorWithDomain:(NSErrorDomain)domain code:(NSInteger)code description:(NSString*)description underlying:(NSError*)underlying {
    // Obj-C throws a fit if there's nulls in dictionaries, so we can't use a dictionary literal here.
    // Use the null-assignment semantics of NSMutableDictionary to make a dictionary either with either, both, or neither key.
    NSMutableDictionary* mut = [[NSMutableDictionary alloc] init];
    mut[NSLocalizedDescriptionKey] = description;
    mut[NSUnderlyingErrorKey] = underlying;

    return [NSError errorWithDomain:domain code:code userInfo:mut];
}

@end

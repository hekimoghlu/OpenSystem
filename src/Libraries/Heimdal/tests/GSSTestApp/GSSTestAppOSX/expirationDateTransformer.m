/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 12, 2025.
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
//  expirationDateTransformer.m
//  GSSTestApp
//
//  Created by Love HÃ¶rnquist Ã…strand on 2013-07-03.
//  Copyright (c) 2013 Apple, Inc. All rights reserved.
//

#import <AppKit/AppKit.h>
#import "expirationDateTransformer.h"

@implementation expirationDateTransformer

+ (Class)transformedValueClass
{
    return [NSString class];
}

+ (BOOL)allowsReverseTransformation
{
    return NO;
}

- (id)transformedValue:(NSDate *)value
{
    
    if (value == nil) return @"expired";
    
    if ([value compare:[NSDate date]] != NSOrderedDescending)
        return @"expired";

    return [NSDateFormatter dateFormatFromTemplate:@"yyyyMMDD HH:MM" options:0 locale:[NSLocale currentLocale]];
}


@end

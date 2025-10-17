/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 29, 2024.
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
#import "config.h"
#import "WKFindConfiguration.h"

#import "WKObject.h"

@implementation WKFindConfiguration

WK_OBJECT_DISABLE_DISABLE_KVC_IVAR_ACCESS;

- (instancetype)init
{
    if (!(self = [super init]))
        return nil;

    _backwards = NO;
    _caseSensitive = NO;
    _wraps = YES;
    return self;
}

- (id)copyWithZone:(NSZone *)zone
{
    WKFindConfiguration *findConfiguration = [(WKFindConfiguration *)[[self class] allocWithZone:zone] init];

    findConfiguration.backwards = _backwards;
    findConfiguration.caseSensitive = _caseSensitive;
    findConfiguration.wraps = _wraps;

    return findConfiguration;
}

@end

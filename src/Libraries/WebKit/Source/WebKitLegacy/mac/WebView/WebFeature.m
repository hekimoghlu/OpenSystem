/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 11, 2021.
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
#import "WebFeatureInternal.h"

@implementation WebFeature

- (instancetype)initWithKey:(NSString *)key preferenceKey:(NSString *)preferenceKey name:(NSString *)name status:(WebFeatureStatus)status category:(WebFeatureCategory)category details:(NSString *)details defaultValue:(BOOL)defaultValue hidden:(BOOL)hidden
{
    if (!(self = [super init]))
        return nil;

    _key = [key copy];
    _preferenceKey = [preferenceKey copy];
    _name = [name copy];
    _status = status;
    _category = category;
    _details = [details copy];
    _defaultValue = defaultValue;
    _hidden = hidden;
    return self;
}

- (void)dealloc
{
    [_key release];
    [_preferenceKey release];
    [_name release];
    [_details release];
    [super dealloc];
}

- (NSString *)description
{
    return [NSString stringWithFormat:@"<%@: %p; name = %@; key = %@; defaultValue = %@>", NSStringFromClass(self.class), self, self.name, self.key, self.defaultValue ? @"on" : @"off"];
}

@end

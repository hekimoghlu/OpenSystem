/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 13, 2022.
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
#import "WKWebProcessBundleParameters.h"

#import <wtf/RetainPtr.h>

@implementation WKWebProcessBundleParameters {
    RetainPtr<NSMutableDictionary> _parameters;
}

- (instancetype)initWithDictionary:(NSDictionary *)dictionary
{
    if (!(self = [super init]))
        return nil;

    _parameters = adoptNS([[NSMutableDictionary alloc] initWithDictionary:dictionary]);

    return self;
}

- (NSString *)description
{
    return [NSString stringWithFormat:@"<%@: %p; parameters = %@>", NSStringFromClass(self.class), self, _parameters.get()];
}

- (NSString *)valueForKey:(NSString *)key
{
    return [_parameters valueForKey:key];
}

- (void)setParameter:(id)parameter forKey:(NSString *)key
{
    [self willChangeValueForKey:key];
    [_parameters setValue:parameter forKey:key];
    [self didChangeValueForKey:key];
}

- (void)setParametersForKeyWithDictionary:(NSDictionary *)dictionary
{
    [dictionary enumerateKeysAndObjectsUsingBlock:^(NSString *key, id parameter, BOOL*) {
        [self setParameter:parameter forKey:key];
    }];
}

@end

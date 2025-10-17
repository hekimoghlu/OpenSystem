/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 31, 2022.
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
#include "config.h"
#include "_WKContentWorldConfiguration.h"

@implementation _WKContentWorldConfiguration {
    String _name;
}

WK_OBJECT_DISABLE_DISABLE_KVC_IVAR_ACCESS;

- (NSString *)name
{
    return _name;
}

- (void)setName:(NSString *)name
{
    _name = name;
}

#pragma mark NSCopying protocol implementation

- (id)copyWithZone:(NSZone *)zone
{
    _WKContentWorldConfiguration *clone = [(_WKContentWorldConfiguration *)[[self class] allocWithZone:zone] init];

    clone.name = self.name;
    clone.allowAccessToClosedShadowRoots = self.allowAccessToClosedShadowRoots;
    clone.allowAutofill = self.allowAutofill;
    clone.allowElementUserInfo = self.allowElementUserInfo;
    clone.disableLegacyBuiltinOverrides = self.disableLegacyBuiltinOverrides;

    return clone;
}

#pragma mark NSSecureCoding protocol implementation

+ (BOOL)supportsSecureCoding
{
    return YES;
}

- (void)encodeWithCoder:(NSCoder *)coder
{
    [coder encodeObject:self.name forKey:@"name"];
    [coder encodeBool:self.allowAccessToClosedShadowRoots forKey:@"allowAccessToClosedShadowRoots"];
    [coder encodeBool:self.allowAutofill forKey:@"allowAutofill"];
    [coder encodeBool:self.allowElementUserInfo forKey:@"allowElementUserInfo"];
    [coder encodeBool:self.disableLegacyBuiltinOverrides forKey:@"disableLegacyBuiltinOverrides"];
}

- (instancetype)initWithCoder:(NSCoder *)coder
{
    if (!(self = [self init]))
        return nil;

    self.name = [coder decodeObjectOfClass:[NSString class] forKey:@"name"];
    self.allowAccessToClosedShadowRoots = [coder decodeBoolForKey:@"allowAccessToClosedShadowRoots"];
    self.allowAutofill = [coder decodeBoolForKey:@"allowAutofill"];
    self.allowElementUserInfo = [coder decodeBoolForKey:@"allowElementUserInfo"];
    self.disableLegacyBuiltinOverrides = [coder decodeBoolForKey:@"disableLegacyBuiltinOverrides"];

    return self;
}

@end

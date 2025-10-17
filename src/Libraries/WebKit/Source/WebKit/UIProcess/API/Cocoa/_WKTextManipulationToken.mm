/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 8, 2024.
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
#import "_WKTextManipulationToken.h"

#import <wtf/RetainPtr.h>

NSString * const _WKTextManipulationTokenUserInfoDocumentURLKey = @"_WKTextManipulationTokenUserInfoDocumentURLKey";
NSString * const _WKTextManipulationTokenUserInfoTagNameKey = @"_WKTextManipulationTokenUserInfoTagNameKey";
NSString * const _WKTextManipulationTokenUserInfoRoleAttributeKey = @"_WKTextManipulationTokenUserInfoRoleAttributeKey";
NSString * const _WKTextManipulationTokenUserInfoVisibilityKey = @"_WKTextManipulationTokenUserInfoVisibilityKey";

@implementation _WKTextManipulationToken {
    RetainPtr<NSDictionary<NSString *, id>> _userInfo;
}

- (void)dealloc
{
    [_identifier release];
    _identifier = nil;
    [_content release];
    _content = nil;

    [super dealloc];
}

- (void)setUserInfo:(NSDictionary<NSString *, id> *)userInfo
{
    if (userInfo == _userInfo || [_userInfo isEqual:userInfo])
        return;

    _userInfo = adoptNS(userInfo.copy);
}

- (NSDictionary<NSString *, id> *)userInfo
{
    return _userInfo.get();
}

static BOOL isEqualOrBothNil(id a, id b)
{
    if (a == b)
        return YES;

    return [a isEqual:b];
}

- (BOOL)isEqual:(id)object
{
    if (self == object)
        return YES;

    if (![object isKindOfClass:[self class]])
        return NO;

    return [self isEqualToTextManipulationToken:object includingContentEquality:YES];
}

- (BOOL)isEqualToTextManipulationToken:(_WKTextManipulationToken *)otherToken includingContentEquality:(BOOL)includingContentEquality
{
    if (!otherToken)
        return NO;

    BOOL equalIdentifiers = isEqualOrBothNil(self.identifier, otherToken.identifier);
    BOOL equalExclusion = self.isExcluded == otherToken.isExcluded;
    BOOL equalContent = !includingContentEquality || isEqualOrBothNil(self.content, otherToken.content);
    BOOL equalUserInfo = isEqualOrBothNil(self.userInfo, otherToken.userInfo);

    return equalIdentifiers && equalExclusion && equalContent && equalUserInfo;
}

- (NSString *)description
{
    return [self _descriptionPreservingPrivacy:YES];
}

- (NSString *)debugDescription
{
    return [self _descriptionPreservingPrivacy:NO];
}

- (NSString *)_descriptionPreservingPrivacy:(BOOL)preservePrivacy
{
    NSMutableString *description = [NSMutableString stringWithFormat:@"<%@: %p; identifier = %@; isExcluded = %i", self.class, self, self.identifier, self.isExcluded];
    if (preservePrivacy)
        [description appendFormat:@"; content length = %lu", (unsigned long)self.content.length];
    else
        [description appendFormat:@"; content = %@; user info = %@", self.content, self.userInfo];

    [description appendString:@">"];

    return adoptNS([description copy]).autorelease();
}

@end

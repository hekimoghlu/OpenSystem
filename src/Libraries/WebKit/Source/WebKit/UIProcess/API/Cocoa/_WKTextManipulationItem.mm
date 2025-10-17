/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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
#import "_WKTextManipulationItem.h"

#import "_WKTextManipulationToken.h"
#import <wtf/RetainPtr.h>

NSString * const _WKTextManipulationItemErrorDomain = @"WKTextManipulationItemErrorDomain";
NSString * const _WKTextManipulationItemErrorItemKey = @"item";

@implementation _WKTextManipulationItem {
    RetainPtr<NSString> _identifier;
    RetainPtr<NSArray<_WKTextManipulationToken *>> _tokens;
}

- (instancetype)initWithIdentifier:(NSString *)identifier tokens:(NSArray<_WKTextManipulationToken *> *)tokens
{
    if (!(self = [super init]))
        return nil;

    _identifier = identifier;
    _tokens = tokens;
    return self;
}

- (instancetype)initWithIdentifier:(NSString *)identifier tokens:(NSArray<_WKTextManipulationToken *> *)tokens isSubframe:(BOOL)isSubframe isCrossSiteSubframe:(BOOL)isCrossSiteSubframe
{
    if (!(self = [super init]))
        return nil;

    _identifier = identifier;
    _tokens = tokens;
    _isSubframe = isSubframe;
    _isCrossSiteSubframe = isCrossSiteSubframe;
    return self;
}

- (NSString *)identifier
{
    return _identifier.get();
}

- (NSArray<_WKTextManipulationToken *> *)tokens
{
    return _tokens.get();
}

- (BOOL)isEqual:(id)object
{
    if (self == object)
        return YES;

    if (![object isKindOfClass:[self class]])
        return NO;

    return [self isEqualToTextManipulationItem:object includingContentEquality:YES];
}

- (BOOL)isEqualToTextManipulationItem:(_WKTextManipulationItem *)otherItem includingContentEquality:(BOOL)includingContentEquality
{
    if (!otherItem)
        return NO;

    if (!(self.identifier == otherItem.identifier || [self.identifier isEqualToString:otherItem.identifier]) || self.tokens.count != otherItem.tokens.count)
        return NO;

    __block BOOL tokensAreEqual = YES;
    NSUInteger count = std::min(self.tokens.count, otherItem.tokens.count);
    [self.tokens enumerateObjectsAtIndexes:[NSIndexSet indexSetWithIndexesInRange:NSMakeRange(0, count)] options:0 usingBlock:^(_WKTextManipulationToken *token, NSUInteger index, BOOL* stop) {
        _WKTextManipulationToken *otherToken = otherItem.tokens[index];
        if (![token isEqualToTextManipulationToken:otherToken includingContentEquality:includingContentEquality]) {
            tokensAreEqual = NO;
            *stop = YES;
        }
    }];

    return tokensAreEqual;
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
    NSMutableArray<NSString *> *recursiveDescriptions = [NSMutableArray array];
    [self.tokens enumerateObjectsUsingBlock:^(_WKTextManipulationToken *token, NSUInteger index, BOOL* stop) {
        NSString *description = preservePrivacy ? token.description : token.debugDescription;
        [recursiveDescriptions addObject:description];
    }];
    NSString *tokenDescription = [NSString stringWithFormat:@"[\n\t%@\n]", [recursiveDescriptions componentsJoinedByString:@",\n\t"]];
    NSString *description = [NSString stringWithFormat:@"<%@: %p; identifier = %@ tokens = %@>", self.class, self, self.identifier, tokenDescription];
    return description;
}

@end

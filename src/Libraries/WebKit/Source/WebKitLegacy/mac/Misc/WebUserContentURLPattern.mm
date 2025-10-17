/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 9, 2025.
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
#import "WebUserContentURLPattern.h"

#import <WebCore/UserContentURLPattern.h>
#import <wtf/URL.h>

using namespace WebCore;

@interface WebUserContentURLPatternPrivate : NSObject
{
@public
    UserContentURLPattern pattern;
}
@end

@implementation WebUserContentURLPatternPrivate
@end

@implementation WebUserContentURLPattern

- (id)initWithPatternString:(NSString *)patternString
{
    self = [super init];
    if (!self)
        return nil;

    _private = [[WebUserContentURLPatternPrivate alloc] init];
    _private->pattern = UserContentURLPattern(String(patternString));

    return self;
}

- (void)dealloc
{
    [_private release];
    _private = nil;

    [super dealloc];
}

- (BOOL)isValid
{
    return _private->pattern.isValid();
}

- (NSString *)scheme
{
    return _private->pattern.scheme();
}

- (NSString *)host
{
    return _private->pattern.host();
}

- (BOOL)matchesSubdomains
{
    return _private->pattern.matchSubdomains();
}

- (BOOL)matchesURL:(NSURL *)url
{
    return _private->pattern.matches(url);
}

@end

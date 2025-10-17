/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 10, 2023.
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
#import <Foundation/Foundation.h>

#if USE(APPLE_INTERNAL_SDK)
#import <Foundation/NSGeometry.h>
#import <Foundation/NSPrivateDecls.h>
#else // USE(APPLE_INTERNAL_SDK)

#if !PLATFORM(MAC) && !PLATFORM(MACCATALYST)
#define NSEDGEINSETS_DEFINED 1
typedef struct NS_SWIFT_SENDABLE NSEdgeInsets {
    CGFloat top;
    CGFloat left;
    CGFloat bottom;
    CGFloat right;
} NSEdgeInsets;
#endif

@interface NSArray ()
- (NSArray *)arrayByExcludingObjectsInArray:(NSArray *)otherArray;
@end

#endif // USE(APPLE_INTERNAL_SDK)

@interface NSTextCheckingResult ()
- (NSDictionary *)detail;
@end

@interface NSHTTPURLResponse ()
+ (BOOL)isErrorStatusCode:(NSInteger)statusCode;
@end

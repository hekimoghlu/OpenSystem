/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 2, 2025.
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
#ifndef SFAnalyticsSampler_Internal_h
#define SFAnalyticsSampler_Internal_h

#if __OBJC2__

#import "SFAnalyticsSampler.h"

NS_ASSUME_NONNULL_BEGIN

@interface SFAnalyticsSampler(Internal)
- (instancetype)initWithName:(NSString*)name interval:(NSTimeInterval)interval block:(NSNumber* (^)(void))block clientClass:(Class)clientClass;
@end

NS_ASSUME_NONNULL_END

#endif // objc2

#endif /* SFAnalyticsSampler_private_h */

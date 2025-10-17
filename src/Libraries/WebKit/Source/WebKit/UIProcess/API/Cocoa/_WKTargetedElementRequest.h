/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 18, 2023.
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
#pragma once

#import <WebKit/WKFoundation.h>

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

WK_CLASS_AVAILABLE(macos(15.0), ios(18.0), visionos(2.0))
@interface _WKTargetedElementRequest : NSObject

- (instancetype)initWithPoint:(CGPoint)point;
- (instancetype)initWithSearchText:(NSString *)searchText;
- (instancetype)initWithSelectors:(NSArray<NSSet<NSString *> *> *)selectors;

@property (nonatomic) BOOL canIncludeNearbyElements;
@property (nonatomic) BOOL shouldIgnorePointerEventsNone;

@end

NS_ASSUME_NONNULL_END

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 17, 2025.
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

@interface ImplClass: NSObject

- (nonnull instancetype)init;

@property (readonly) int letProperty1;
@property (assign) int implProperty;

- (void)mainMethod:(int)param;

@end


@interface ImplClass (Category1)

- (void)category1Method:(int)param;

@end


@interface ImplClass (Category2)

- (void)category2Method:(int)param;

@end


@interface NoImplClass

- (void)noImplMethod:(int)param;

@end

@interface NoInitImplClass: NSObject

@property (readonly, strong, nonnull) NSString *s1;
@property (strong, nonnull) NSString *s2;
@property (readonly, strong, nonnull) NSString *s3;
@property (strong, nonnull) NSString *s4;

@end

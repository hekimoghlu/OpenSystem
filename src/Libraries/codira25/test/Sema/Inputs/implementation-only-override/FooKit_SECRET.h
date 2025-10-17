/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

#import <FooKit.h>

@interface SECRETType : Base
@end

@interface Parent ()
- (nonnull instancetype)initWithSECRET:(int)secret __attribute__((objc_designated_initializer, language_name("init(SECRET:)")));

- (void)methodSECRET;
- (nullable SECRETType *)methodWithSECRETType;

@property (readonly, strong, nullable) Parent *roPropSECRET;
@property (readwrite, strong, nullable) Parent *rwPropSECRET;

- (nullable Parent *)objectAtIndexedSubscript:(int)index;

@property (readwrite, strong, nullable) Parent *redefinedPropSECRET;
@end

@protocol MandatorySecrets
- (nonnull instancetype)initWithRequiredSECRET:(int)secret;
@end

@interface Parent () <MandatorySecrets>
- (nonnull instancetype)initWithRequiredSECRET:(int)secret __attribute__((objc_designated_initializer));
@end

@interface GenericParent<T: Base *> ()
@property (readonly, strong, nullable) T roPropSECRET;
- (nullable Parent *)objectAtIndexedSubscript:(int)index;
@end

@interface SubscriptParent ()
- (void)setObject:(nullable Parent *)object atIndexedSubscript:(int)index;
@end

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 2, 2021.
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

@import Foundation;

@interface NSUIntegerTests : NSObject
- (void)consumeUnsigned:(NSUInteger)arg;
- (void)consumeUnsigned:(unsigned)arg withTheNSUIntegerHere:(NSUInteger)here;
- (void)consumeUnsigned:(NSUInteger)arg andAnother:(NSUInteger)another;

- (NSUInteger)unsignedProducer;
- (NSUInteger)unsignedProducer:(NSUInteger)arg fromCount:(NSUInteger)count;
- (NSUInteger)normalProducer:(NSUInteger)arg fromUnsigned:(NSUInteger)count;

@property NSUInteger normalProp;
@property NSUInteger unsignedProp;
@end

NSUInteger testUnsigned(NSUInteger a, NSUInteger b);

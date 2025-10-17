/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 23, 2025.
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

#include <Foundation/Foundation.h>

@interface Butt: NSObject

- (instancetype _Nonnull)init;

- (void)butt:(NSInteger)x completionHandler:(void (^ _Nonnull)(NSInteger))handler;

@end

@interface MutableButt: Butt
@end

@interface MutableButt_2Fast2Furious: MutableButt
@end

@interface Farm: NSObject

-(void)getDogWithCompletion:(void (^ _Nonnull)(NSInteger))completionHandler
  __attribute__((language_async_name("getter:doggo()")));

-(void)obtainCat:(void (^ _Nonnull)(NSInteger, NSError* _Nullable))completionHandler
__attribute__((language_async_name("getter:catto()")));

@end

void scheduleButt(Butt *b, NSString *s);

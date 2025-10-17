/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 29, 2024.
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

#import "Escaper.h"

#import "DangerousEscaper.h"

@interface Escaper () <DangerousEscaper>
  -(void)mightBeNaughty:(void (^)(void))completion;
@end

@implementation Escaper

- (id)init {
  if ((self = [super init]) != nil) {
    self.escape = ^{};
  }
  return self;
}

-(void)mightBeNaughty:(void (^)(void))completion {
  self.escape = completion;
  completion();
}

@end

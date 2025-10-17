/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 15, 2023.
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

#include "objc_async.h"
#include <stdio.h>

@implementation Butt

- (instancetype)init {
  return [super init];
}

- (void)butt:(NSInteger)x completionHandler:(void (^)(NSInteger))handler {
  printf("starting %ld\n", (long)x);
  handler(679);
}

@end

@implementation MutableButt: Butt
@end

@implementation MutableButt_2Fast2Furious: MutableButt
@end


@implementation Farm

-(void)getDogWithCompletion:(void (^ _Nonnull)(NSInteger))completionHandler {
  printf("getting dog\n");
  completionHandler(123);
}

-(void)obtainCat:(void (^ _Nonnull)(NSInteger, NSError* _Nullable))completionHandler {
  printf("obtaining cat has failed!\n");
  completionHandler(nil, [NSError errorWithDomain:@"obtainCat" code:456 userInfo:nil]);
}

@end

void scheduleButt(Butt *b, NSString *s) {
  [b butt: 1738 completionHandler: ^(NSInteger i) {
    printf("butt %p named %s occurred at %zd\n", b, [s UTF8String], (ssize_t)i);
    fflush(stdout);
  }];
}

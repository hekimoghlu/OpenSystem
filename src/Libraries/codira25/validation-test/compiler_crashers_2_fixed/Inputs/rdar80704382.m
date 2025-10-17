/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 5, 2024.
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

#include "rdar80704382.h"

#pragma clang assume_nonnull begin

@implementation PFXObject
- (instancetype)init {
  if (self = [super init]) {
  }
  return self;
}
+ (void)getIdentifierForUserVisibleFileAtURL:(NSURL *)url
                           completionHandler:
                               (void (^)(FileProviderItemIdentifier __nullable
                                             itemIdentifier,
                                         FileProviderDomainIdentifier __nullable
                                             domainIdentifier,
                                         NSError *__nullable error))
                                   completionHandler {
  completionHandler(@"item_id", @"file_id", NULL);
}
@end

#pragma clang assume_nonnull end

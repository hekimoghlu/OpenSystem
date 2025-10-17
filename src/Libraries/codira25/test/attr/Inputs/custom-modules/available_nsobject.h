/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 19, 2025.
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

__attribute__((availability(macosx,introduced=10.0)))
__attribute__((availability(ios,introduced=2.0)))
__attribute__((availability(tvos,introduced=1.0)))
__attribute__((availability(watchos,introduced=2.0)))
__attribute__((availability(maccatalyst,introduced=13.1)))
@interface NSBaseClass : NSObject
- (instancetype) init
  __attribute__((objc_designated_initializer))
  __attribute__((availability(macosx,introduced=10.0)))
  __attribute__((availability(ios,introduced=2.0)))
  __attribute__((availability(tvos,introduced=1.0)))
  __attribute__((availability(watchos,introduced=2.0)))
  __attribute__((availability(maccatalyst,introduced=13.1)));
@end

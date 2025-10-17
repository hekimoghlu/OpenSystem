/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 27, 2025.
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

#pragma clang assume_nonnull begin

#define LANGUAGE_MAIN_ACTOR __attribute__((language_attr("@MainActor")))
#define LANGUAGE_UI_ACTOR __attribute__((language_attr("@UIActor")))

// NOTE: If you ever end up removing support for the "@UIActor" alias,
// just change both to be @MainActor and it won't change the purpose of
// this test.

LANGUAGE_UI_ACTOR LANGUAGE_MAIN_ACTOR @protocol DoubleMainActor
@required
- (NSString *)createSeaShanty:(NSInteger)number;
@end

#pragma clang assume_nonnull end

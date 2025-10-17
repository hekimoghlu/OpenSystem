/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 24, 2022.
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

#if __LANGUAGE_ATTR_SUPPORTS_MACROS
#define ADD_ASYNC __attribute__((language_attr("@macro_library.AddAsync")))
#define ADD_ASYNC_FINAL __attribute__((language_attr("@macro_library.AddAsyncFinal")))
#define DO_SOMETHING_DOTTED __attribute__((language_attr("@AcceptDotted(.something)")))
#else
#define ADD_ASYNC
#define ADD_ASYNC_FINAL
#define DO_SOMETHING_DOTTED
#endif

void async_divide(double x, double y, void (^ _Nonnull completionHandler)(double x)) ADD_ASYNC;

typedef struct SlowComputer {
} SlowComputer;

void computer_divide(const SlowComputer *computer, double x, double y, void (^ _Nonnull completionHandler)(double x))
  ADD_ASYNC
  __attribute__((language_name("SlowComputer.divide(self:_:_:completionHandler:)")));

void f1(double x) DO_SOMETHING_DOTTED;
void f2(double x) DO_SOMETHING_DOTTED;
void f3(double x) DO_SOMETHING_DOTTED;

#if __OBJC__
@import Foundation;

@interface Computer: NSObject
-(void)multiply:(double)x by:(double)y afterDone:(void (^ _Nonnull)(double x))afterDone
  ADD_ASYNC_FINAL
  __attribute__((language_async(none)));
@end
#endif

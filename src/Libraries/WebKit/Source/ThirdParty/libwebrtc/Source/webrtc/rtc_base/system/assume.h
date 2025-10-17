/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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
#ifndef RTC_BASE_SYSTEM_ASSUME_H_
#define RTC_BASE_SYSTEM_ASSUME_H_

// Possibly evaluate `p`, promising the compiler that the result is true; the
// compiler is allowed (but not required) to use this information when
// optimizing the code. USE WITH CAUTION! If you promise the compiler things
// that aren't true, it will build a broken binary for you.
//
// As a simple example, the compiler is allowed to transform this
//
//   RTC_ASSUME(x == 4);
//   return x;
//
// into this
//
//   return 4;
//
// It is even allowed to propagate the assumption "backwards in time", if it can
// prove that it must have held at some earlier time. For example, the compiler
// is allowed to transform this
//
//   int Add(int x, int y) {
//     if (x == 17)
//       y += 1;
//     RTC_ASSUME(x != 17);
//     return x + y;
//   }
//
// into this
//
//   int Add(int x, int y) {
//     return x + y;
//   }
//
// since if `x` isn't 17 on the third line of the function body, the test of `x
// == 17` on the first line must fail since nothing can modify the local
// variable `x` in between.
//
// The intended use is to allow the compiler to optimize better. For example,
// here we allow the compiler to omit an instruction that ensures correct
// rounding of negative arguments:
//
//   int DivBy2(int x) {
//     RTC_ASSUME(x >= 0);
//     return x / 2;
//   }
//
// and here we allow the compiler to possibly omit a null check:
//
//   void Delete(int* p) {
//     RTC_ASSUME(p != nullptr);
//     delete p;
//   }
//
// clang-format off
#if defined(__GNUC__)
#define RTC_ASSUME(p) do { if (!(p)) __builtin_unreachable(); } while (0)
#else
#define RTC_ASSUME(p) do {} while (0)
#endif
// clang-format on

#endif  // RTC_BASE_SYSTEM_ASSUME_H_

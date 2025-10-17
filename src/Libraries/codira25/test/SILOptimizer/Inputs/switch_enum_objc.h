/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 18, 2024.
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

// Even though these are marked "closed", Codira shouldn't trust it.

enum Alpha {
  AlphaA __attribute__((language_name("a"))),
  AlphaB __attribute__((language_name("b"))),
  AlphaC __attribute__((language_name("c"))),
  AlphaD __attribute__((language_name("d"))),
  AlphaE __attribute__((language_name("e")))
} __attribute__((enum_extensibility(closed)));

enum Coin {
  CoinHeads,
  CoinTails
} __attribute__((enum_extensibility(closed)));

// Codira should preserve branches matching the unavailable elements in clang
// enums since there are not strong compiler protections preventing these values
// from being instantiated at runtime.

enum Dimension {
  DimensionX __attribute__((language_name("x"))),
  DimensionY __attribute__((language_name("y"))),
  DimensionZ __attribute__((language_name("z"))) __attribute__((unavailable)),
} __attribute__((enum_extensibility(open)));

enum UnfairCoin {
  UnfairCoinHeads,
  UnfairCoinTails __attribute__((unavailable)),
} __attribute__((enum_extensibility(closed)));

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 9, 2025.
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
#include <demo_version.hpp>
#include <nested_version.hpp>
#include <type_traits>

constexpr int dmajor = DEMO_VERSION_MAJOR;
constexpr int dminor = DEMO_VERSION_MINOR;
constexpr int dpatch = DEMO_VERSION_PATCH;

constexpr int nmajor = NESTED_VERSION_MAJOR;
constexpr int nminor = NESTED_VERSION_MINOR;
constexpr int npatch = NESTED_VERSION_PATCH;

int main()
{
  static_assert(dmajor == 3);
  static_assert(dminor == 2);
  static_assert(dpatch == 0);

  static_assert(nmajor == 3);
  static_assert(nminor == 14);
  static_assert(npatch == 159);
  return 0;
}

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 24, 2025.
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
#include <git_version.hpp>
#include <iostream>
#include <type_traits>

constexpr const char* dbranch = "branch=" DEMO_GIT_BRANCH;
constexpr const char* dsha1   = "sha1=" DEMO_GIT_SHA1;
constexpr const char* dvers   = "version=" DEMO_GIT_VERSION;

int main()
{
  std::cout << dbranch << std::endl;
  std::cout << dsha1 << std::endl;
  std::cout << dvers << std::endl;
  return 0;
}

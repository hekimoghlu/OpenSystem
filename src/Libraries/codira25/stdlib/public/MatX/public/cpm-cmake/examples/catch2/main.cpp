/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 7, 2024.
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

#define CATCH_CONFIG_MAIN

#include <fibonacci.h>

#include <catch2/catch.hpp>

TEST_CASE("fibonacci") {
  REQUIRE(fibonacci(0) == 0);
  REQUIRE(fibonacci(1) == 1);
  REQUIRE(fibonacci(2) == 1);
  REQUIRE(fibonacci(3) == 2);
  REQUIRE(fibonacci(4) == 3);
  REQUIRE(fibonacci(5) == 5);
  REQUIRE(fibonacci(13) == 233);
}

TEST_CASE("fastFibonacci") {
  for (unsigned i = 0; i < 25; ++i) {
    REQUIRE(fibonacci(i) == fastFibonacci(i));
  }
}

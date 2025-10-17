/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 2, 2022.
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

#include <catch2/catch.hpp>
#include <iostream>

#include "nvexec/detail/memory.cuh"
#include "tracer_resource.h"

namespace {

  TEST_CASE("monotonic buffer releases storage", "[cuda][stream][memory][monotonic buffer]") {
    tracer_resource resource{};

    {
      nvdetail::monotonic_buffer_resource buffer{1024, &resource};

      void* ptr_1 = buffer.allocate(128, 8);
      void* ptr_2 = buffer.allocate(256, 16);
      REQUIRE(ptr_1 != nullptr);
      REQUIRE(ptr_2 != nullptr);
      REQUIRE(ptr_1 != ptr_2);
      REQUIRE(1 == resource.allocations.size());

      buffer.deallocate(ptr_1, 128, 8);
      buffer.deallocate(ptr_2, 256, 16);
      REQUIRE(1 == resource.allocations.size());
    }

    REQUIRE(0 == resource.allocations.size());
  }

  TEST_CASE(
    "monotonic buffer keeps track of new allocations",
    "[cuda][stream][memory][monotonic buffer]") {
    tracer_resource resource{};

    {
      nvdetail::monotonic_buffer_resource buffer{1024, &resource};

      void* ptr_1 = buffer.allocate(128, 8);
      void* ptr_2 = buffer.allocate(256, 16);
      void* ptr_3 = buffer.allocate(1024, 1);
      void* ptr_4 = buffer.allocate(512, 2);
      REQUIRE(ptr_1 != nullptr);
      REQUIRE(ptr_2 != nullptr);
      REQUIRE(ptr_3 != nullptr);
      REQUIRE(ptr_4 != nullptr);
      REQUIRE(ptr_1 != ptr_2);
      REQUIRE(ptr_2 != ptr_3);
      REQUIRE(ptr_3 != ptr_4);
      REQUIRE(2 == resource.allocations.size());

      buffer.deallocate(ptr_1, 128, 8);
      buffer.deallocate(ptr_2, 256, 16);
      buffer.deallocate(ptr_3, 1024, 1);
      buffer.deallocate(ptr_4, 512, 2);
      REQUIRE(2 == resource.allocations.size());
    }

    REQUIRE(0 == resource.allocations.size());
  }

  TEST_CASE(
    "monotonic buffer provides required allocations",
    "[cuda][stream][memory][monotonic buffer]") {
    tracer_resource resource{};
    nvdetail::monotonic_buffer_resource buffer{1024, &resource};

    char* ptr_1 = reinterpret_cast<char*>(buffer.allocate(32, 8));
    char* ptr_2 = reinterpret_cast<char*>(buffer.allocate(32, 8));

    size_t distance = std::distance(ptr_1, ptr_2);
    REQUIRE(distance == 32);

    buffer.deallocate(ptr_1, 32, 8);
    buffer.deallocate(ptr_2, 32, 16);
  }

  TEST_CASE(
    "monotonic buffer provides required alignment",
    "[cuda][stream][memory][monotonic buffer]") {
    tracer_resource resource{};
    nvdetail::monotonic_buffer_resource buffer{2048, &resource};

    for (int alignment = 1; alignment < 512; alignment *= 2) {
      void* ptr = buffer.allocate(32, alignment);
      REQUIRE(reinterpret_cast<uintptr_t>(ptr) % alignment == 0);
      buffer.deallocate(ptr, 32, alignment);
    }
  }
} // namespace

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

#include <catch2/catch.hpp>
#include <iostream>

#include "nvexec/detail/memory.cuh"
#include "tracer_resource.h"

namespace {

  TEST_CASE("synchronized pool releases storage", "[cuda][stream][memory][synchronized pool]") {
    tracer_resource resource{};

    {
      nvdetail::synchronized_pool_resource pool{&resource};

      void* ptr_1 = pool.allocate(128, 8);
      void* ptr_2 = pool.allocate(256, 16);
      REQUIRE(ptr_1 != nullptr);
      REQUIRE(ptr_2 != nullptr);
      REQUIRE(ptr_1 != ptr_2);
      REQUIRE(2 == resource.allocations.size());

      pool.deallocate(ptr_2, 256, 16);
      pool.deallocate(ptr_1, 128, 8);
      REQUIRE(2 == resource.allocations.size());
    }

    REQUIRE(0 == resource.allocations.size());
  }

  TEST_CASE("synchronized pool caches allocations", "[cuda][stream][memory][synchronized pool]") {
    tracer_resource resource{};

    {
      nvdetail::synchronized_pool_resource pool{&resource};

      for (int i = 0; i < 10; i++) {
        void* ptr_1 = pool.allocate(128, 8);
        void* ptr_2 = pool.allocate(256, 16);
        REQUIRE(ptr_1 != nullptr);
        REQUIRE(ptr_2 != nullptr);
        REQUIRE(ptr_1 != ptr_2);
        REQUIRE(2 == resource.allocations.size());

        pool.deallocate(ptr_2, 256, 16);
        pool.deallocate(ptr_1, 128, 8);
        REQUIRE(2 == resource.allocations.size());
      }

      REQUIRE(2 == resource.allocations.size());
    }

    REQUIRE(0 == resource.allocations.size());
  }

  TEST_CASE(
    "synchronized pool doesn't touch allocated memory",
    "[cuda][stream][memory][synchronized pool]") {
    tracer_resource resource{};
    nvdetail::synchronized_pool_resource pool{&resource};

    for (int n = 32; n < 512; n *= 2) {
      int bytes = n * sizeof(int);
      int alignment = alignof(int);

      int* ptr = reinterpret_cast<int*>(pool.allocate(bytes, alignment));
      std::iota(ptr, ptr + n, n);
      pool.deallocate(ptr, 128, 8);

      ptr = reinterpret_cast<int*>(pool.allocate(bytes, alignment));
      for (int i = 0; i < n; i++) {
        REQUIRE(ptr[i] == i + n);
      }
      pool.deallocate(ptr, 128, 8);
    }
  }

  TEST_CASE(
    "synchronized pool provides required alignment",
    "[cuda][stream][memory][synchronized pool]") {
    tracer_resource resource{};
    nvdetail::synchronized_pool_resource pool{&resource};

    for (int alignment = 1; alignment < 512; alignment *= 2) {
      void* ptr = pool.allocate(32, alignment);
      INFO("Alignment: " << alignment);
      REQUIRE(reinterpret_cast<uintptr_t>(ptr) % alignment == 0);
      pool.deallocate(ptr, 32, alignment);
    }
  }
} // namespace

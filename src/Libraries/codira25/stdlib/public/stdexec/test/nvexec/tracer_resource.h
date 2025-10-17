/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 7, 2022.
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

#pragma once

#include <new>
#include <vector>
#include <memory_resource>

#include <catch2/catch.hpp>

namespace nvdetail = nvexec::_strm;

namespace {

  struct tracer_resource : public std::pmr::memory_resource {
    struct allocation_info_t {
      void* ptr;
      size_t bytes;
      size_t alignment;

      bool operator==(const allocation_info_t& other) const noexcept {
        return ptr == other.ptr && bytes == other.bytes && alignment == other.alignment;
      }
    };

    std::vector<allocation_info_t> allocations;

    void* do_allocate(size_t bytes, size_t alignment) override {
      INFO("Allocate: " << bytes << " bytes, " << alignment << " alignment");
      void* ptr = ::operator new[](bytes, std::align_val_t(alignment));
      allocations.push_back(allocation_info_t{ptr, bytes, alignment});
      return ptr;
    }

    void do_deallocate(void* ptr, size_t bytes, size_t alignment) override {
      INFO("Deallocate: " << bytes << " bytes, " << alignment << " alignment");

      auto it =
        std::find(allocations.begin(), allocations.end(), allocation_info_t{ptr, bytes, alignment});

      REQUIRE(it != allocations.end());

      if (it != allocations.end()) {
        allocations.erase(it);
        ::operator delete[](ptr, std::align_val_t(alignment));
      }
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
      return this == &other;
    }
  };
} // namespace
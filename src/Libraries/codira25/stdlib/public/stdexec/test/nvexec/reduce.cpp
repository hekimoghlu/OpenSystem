/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 25, 2025.
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

#include <stdexec/execution.hpp>

#include "nvexec/stream_context.cuh"
#include "common.cuh"

#include <thrust/device_vector.h>
#include <cub/thread/thread_operators.cuh>

#include <algorithm>
#include <span>

namespace ex = stdexec;

namespace {
  struct minimum {
    template <class T1, class T2>
    constexpr auto
      operator()(const T1& lhs, const T2& rhs) const -> _CUDA_VSTD::common_type_t<T1, T2> {
      return (lhs < rhs) ? lhs : rhs;
    }
  };

  TEST_CASE(
    "nvexec reduce returns a sender with single input",
    "[cuda][stream][adaptors][reduce]") {
    constexpr int N = 2048;
    int input[N] = {};
    std::fill_n(input, N, 1);

    nvexec::stream_context stream{};
    auto snd = ex::transfer_just(stream.get_scheduler(), std::span{input}) | nvexec::reduce(0);

    STATIC_REQUIRE(ex::sender_of<decltype(snd), ex::set_value_t(int&)>);

    (void) snd;
  }

  TEST_CASE("nvexec reduce returns a sender with two inputs", "[cuda][stream][adaptors][reduce]") {
    constexpr int N = 2048;
    int input[N] = {};
    std::fill_n(input, N, 1);

    nvexec::stream_context stream{};
    auto snd = ex::transfer_just(stream.get_scheduler(), std::span{input})
             | nvexec::reduce(0, cuda::std::plus{});

    STATIC_REQUIRE(ex::sender_of<decltype(snd), ex::set_value_t(int&)>);

    (void) snd;
  }

  TEST_CASE("nvexec reduce uses sum as default", "[cuda][stream][adaptors][reduce]") {
    constexpr int N = 2048;
    constexpr int init = 42;

    thrust::device_vector<int> input(N, 1);
    int* first = thrust::raw_pointer_cast(input.data());
    int* last = thrust::raw_pointer_cast(input.data()) + input.size();

    nvexec::stream_context stream{};
    auto snd = ex::transfer_just(stream.get_scheduler(), std::span{first, last})
             | nvexec::reduce(init);

    auto [result] = ex::sync_wait(std::move(snd)).value();

    REQUIRE(result == N + init);
  }

  TEST_CASE("nvexec reduce uses the passed function", "[cuda][stream][adaptors][reduce]") {
    constexpr int N = 2048;
    constexpr int init = 42;

    thrust::device_vector<int> input(N, 1);
    int* first = thrust::raw_pointer_cast(input.data());
    int* last = thrust::raw_pointer_cast(input.data()) + input.size();

    nvexec::stream_context stream{};
    auto snd = ex::transfer_just(stream.get_scheduler(), std::span{first, last})
             | nvexec::reduce(init, minimum{});

    auto [result] = ex::sync_wait(std::move(snd)).value();

    REQUIRE(result == 1);
  }

  TEST_CASE("nvexec reduce executes on GPU", "[cuda][stream][adaptors][reduce]") {
    constexpr int N = 2048;
    constexpr int init = 42;

    thrust::device_vector<int> input(N, 1);
    int* first = thrust::raw_pointer_cast(input.data());
    int* last = thrust::raw_pointer_cast(input.data()) + input.size();

    auto is_on_gpu = [](const int left, const int right) {
      return nvexec::is_on_gpu() ? left + right : 0;
    };

    nvexec::stream_context stream{};
    auto snd = ex::transfer_just(stream.get_scheduler(), std::span{first, last})
             | nvexec::reduce(init, is_on_gpu);

    auto [result] = ex::sync_wait(std::move(snd)).value();

    REQUIRE(result == N + init);
  }
} // namespace

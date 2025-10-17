/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 3, 2023.
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

#include <cub/config.cuh>

#include <thrust/detail/algorithm_wrapper.h>

#include <format>
#include <string_view>

#include <nlohmann/json.hpp>

CUB_NAMESPACE_BEGIN

namespace detail::ptx_json
{
inline nlohmann::json parse(std::string_view tag, std::string_view ptx_stream)
{
  auto const open_tag      = std::format("cccl.ptx_json.begin({})", tag);
  auto const open_location = std::ranges::search(ptx_stream, open_tag);
  if (std::ranges::size(open_location) != open_tag.size())
  {
    return nullptr;
  }

  auto const close_tag      = std::format("cccl.ptx_json.end({})", tag);
  auto const close_location = std::ranges::search(ptx_stream, close_tag);
  if (std::ranges::size(close_location) != close_location.size())
  {
    return nullptr;
  }

  return nlohmann::json::parse(std::ranges::end(open_location), std::ranges::begin(close_location), nullptr, true, true);
}
} // namespace detail::ptx_json

CUB_NAMESPACE_END

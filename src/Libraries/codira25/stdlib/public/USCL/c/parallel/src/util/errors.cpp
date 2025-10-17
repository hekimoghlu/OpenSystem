/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 14, 2021.
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
#include "errors.h"

#include <stdexcept>

void check(nvrtcResult result)
{
  if (result != NVRTC_SUCCESS)
  {
    throw std::runtime_error(std::string("NVRTC error: ") + nvrtcGetErrorString(result));
  }
}

void check(CUresult result)
{
  if (result != CUDA_SUCCESS)
  {
    const char* str = nullptr;
    cuGetErrorString(result, &str);
    throw std::runtime_error(std::string("CUDA error: ") + str);
  }
}

void check(nvJitLinkResult result)
{
  if (result != NVJITLINK_SUCCESS)
  {
    throw std::runtime_error(std::string("nvJitLink error: ") + std::to_string(result));
  }
}

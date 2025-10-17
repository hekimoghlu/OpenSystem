/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 4, 2023.
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

#include <thrust/execution_policy.h>

#include <c2h/checked_allocator.cuh>

namespace c2h
{
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
static const auto device_policy        = THRUST_NS_QUALIFIER::cuda::par(checked_cuda_allocator<char>{});
static const auto nosync_device_policy = THRUST_NS_QUALIFIER::cuda::par_nosync(checked_cuda_allocator<char>{});
#else // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
static const auto device_policy        = THRUST_NS_QUALIFIER::device;
static const auto nosync_device_policy = THRUST_NS_QUALIFIER::device;
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

} // namespace c2h

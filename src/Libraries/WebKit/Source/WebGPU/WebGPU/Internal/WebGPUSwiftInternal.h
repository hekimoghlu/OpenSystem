/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 17, 2023.
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

#include "APIConversions.h"
#include "Buffer.h"
#include "CommandEncoder.h"
#include "CommandsMixin.h"
#include "ComputePassEncoder.h"
#include "Device.h"
#include "IsValidToUseWith.h"
#include "QuerySet.h"
#include "Queue.h"
#include "RenderPassEncoder.h"
#include "Texture.h"
#include "TextureView.h"
#include "WebGPU.h"
#include <cstdint>
#include <span>
#include <wtf/CheckedArithmetic.h>
#include <wtf/HashSet.h>
#include <wtf/Ref.h>
#include <wtf/StdLibExtras.h>

using SpanConstUInt8 = std::span<const uint8_t>;
using SpanUInt8 = std::span<uint8_t>;
using WTFRangeSizeT = WTF::Range<size_t>;

__attribute__((used)) static const auto stdDynamicExtent = std::dynamic_extent;

// FIXME: importing WTF::Range does not work
namespace WTF {
template<typename PassedType>
class Range;
}
using RefComputePassEncoder = Ref<WebGPU::ComputePassEncoder>;
inline unsigned long roundUpToMultipleOfNonPowerOfTwoCheckedUInt32UnsignedLong(Checked<uint32_t> x, unsigned long y) { return WTF::roundUpToMultipleOfNonPowerOfTwo<unsigned long int, Checked<uint32_t>>(x, y); }
inline uint32_t roundUpToMultipleOfNonPowerOfTwoUInt32UInt32(uint32_t a, uint32_t b) { return WTF::roundUpToMultipleOfNonPowerOfTwo<uint32_t, Checked<uint32_t>>(a, b); }

// FIXME: rdar://140819194
constexpr unsigned long int WGPU_COPY_STRIDE_UNDEFINED_ = WGPU_COPY_STRIDE_UNDEFINED;

// FIXME: rdar://140819448
constexpr auto MTLBlitOptionNone_ = MTLBlitOptionNone;

inline Checked<size_t> checkedDifferenceSizeT(size_t left, size_t right)
{
    return WTF::checkedDifference<size_t>(left, right);
}

using RefRenderPassEncoder = Ref<WebGPU::RenderPassEncoder>;
using SliceSet = HashSet<uint64_t, DefaultHash<uint64_t>, WTF::UnsignedWithZeroKeyHashTraits<uint64_t>>;
inline bool isValidToUseWithTextureViewCommandEncoder(const WebGPU::TextureView& texture, const WebGPU::CommandEncoder& commandEncoder)
{
    return WebGPU::isValidToUseWith(texture, commandEncoder);
}

inline double clampDouble(const double& v, const double& lo, const double& hi)
{
    return std::clamp(v, lo, hi);
}

#ifndef __swift__
#include "WebGPUSwift-Generated.h"
#endif

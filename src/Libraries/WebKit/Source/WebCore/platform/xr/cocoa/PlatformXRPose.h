/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 23, 2025.
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

#if ENABLE(WEBXR) && PLATFORM(COCOA)

#include <WebCore/PlatformXR.h>
#include <simd/simd.h>
#include <wtf/TZoneMalloc.h>

class PlatformXRPose {
    WTF_MAKE_TZONE_ALLOCATED(PlatformXRPose);

public:
    simd_float4x4 simdTransform() const { return m_simdTransform; }
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    simd_float3 simdPosition() const { return m_simdTransform.columns[3].xyz; }
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
    simd_quatf simdOrientation() const { return simd_quaternion(m_simdTransform); }
    WEBCORE_EXPORT WebCore::FloatPoint3D position() const;
    WEBCORE_EXPORT PlatformXR::FrameData::FloatQuaternion orientation() const;
    WEBCORE_EXPORT PlatformXR::FrameData::Pose pose() const;

    using FloatMatrix4 = std::array<float, 16>;
    WEBCORE_EXPORT FloatMatrix4 toColumnMajorFloatArray() const;

    WEBCORE_EXPORT float distanceToPose(const PlatformXRPose&) const;
    WEBCORE_EXPORT PlatformXRPose verticalTransformPose() const;

    WEBCORE_EXPORT PlatformXRPose(const simd_float4x4&);
    WEBCORE_EXPORT PlatformXRPose(const simd_float4x4&, const simd_float4x4& parentTransform);

private:
    simd_float4x4 m_simdTransform;
};

#endif // ENABLE(WEBXR) && PLATFORM(COCOA)

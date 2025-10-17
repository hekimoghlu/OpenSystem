/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 27, 2023.
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

namespace WTF {
class ASCIILiteral;
class PrintStream;
class String;
}

namespace WGSL {

#define ENUM_AddressSpace(value) \
    value(Function, function) \
    value(Handle, handle) \
    value(Private, private) \
    value(Storage, storage) \
    value(Uniform, uniform) \
    value(Workgroup, workgroup) \

#define ENUM_AccessMode(value) \
    value(Read, read) \
    value(ReadWrite, read_write) \
    value(Write, write) \

#define ENUM_TexelFormat(value) \
    value(BGRA8unorm, bgra8unorm) \
    value(R32float, r32float) \
    value(R32sint, r32sint) \
    value(R32uint, r32uint) \
    value(RG32float, rg32float) \
    value(RG32sint, rg32sint) \
    value(RG32uint, rg32uint) \
    value(RGBA16float, rgba16float) \
    value(RGBA16sint, rgba16sint) \
    value(RGBA16uint, rgba16uint) \
    value(RGBA32float, rgba32float) \
    value(RGBA32sint, rgba32sint) \
    value(RGBA32uint, rgba32uint) \
    value(RGBA8sint, rgba8sint) \
    value(RGBA8snorm, rgba8snorm) \
    value(RGBA8uint, rgba8uint) \
    value(RGBA8unorm, rgba8unorm) \

#define ENUM_InterpolationType(value) \
    value(Flat, flat) \
    value(Linear, linear) \
    value(Perspective, perspective)

#define ENUM_InterpolationSampling(value) \
    value(Center, center) \
    value(Centroid, centroid) \
    value(Either, either) \
    value(First, first) \
    value(Sample, sample)

#define ENUM_ShaderStage(value) \
    value(Compute,  compute,  1 << 2) \
    value(Fragment, fragment, 1 << 1) \
    value(Vertex,   vertex,   1 << 0) \

#define ENUM_SeverityControl(value) \
    value(Error, error) \
    value(Info, info) \
    value(Off, off) \
    value(Warning, warning) \

#define ENUM_Builtin(value) \
    value(FragDepth, frag_depth) \
    value(FrontFacing, front_facing) \
    value(GlobalInvocationId, global_invocation_id) \
    value(InstanceIndex, instance_index) \
    value(LocalInvocationId, local_invocation_id) \
    value(LocalInvocationIndex, local_invocation_index) \
    value(NumWorkgroups, num_workgroups) \
    value(Position, position) \
    value(SampleIndex, sample_index) \
    value(SampleMask, sample_mask) \
    value(VertexIndex, vertex_index) \
    value(WorkgroupId, workgroup_id) \

#define ENUM_Extension(value) \
    value(F16, f16, 1 << 0) \

#define ENUM_LanguageFeature(value) \
    value(Packed4x8IntegerDotProduct, packed_4x8_integer_dot_product, 1 << 0) \
    value(PointerCompositeAccess, pointer_composite_access, 1 << 1) \
    value(ReadonlyAndReadwriteStorageTextures, readonly_and_readwrite_storage_textures, 1 << 2) \
    value(UnrestrictedPointerParameters, unrestricted_pointer_parameters, 1 << 3) \

#define ENUM_DECLARE_VALUE(__value, _, ...) \
    __value __VA_OPT__(=) __VA_ARGS__,

#define ENUM_DECLARE_PRINT_INTERNAL(__name) \
    void printInternal(WTF::PrintStream& out, __name)

#define ENUM_DECLARE_TO_STRING(__name) \
    WTF::ASCIILiteral toString(__name)

#define ENUM_DECLARE_PARSE(__name) \
    const __name* parse##__name(const WTF::String&)

#define ENUM_DECLARE(__name) \
    enum class __name : uint8_t { \
    ENUM_##__name(ENUM_DECLARE_VALUE) \
    }; \
    ENUM_DECLARE_PRINT_INTERNAL(__name); \
    ENUM_DECLARE_TO_STRING(__name); \
    ENUM_DECLARE_PARSE(__name);

ENUM_DECLARE(AddressSpace);
ENUM_DECLARE(AccessMode);
ENUM_DECLARE(TexelFormat);
ENUM_DECLARE(InterpolationType);
ENUM_DECLARE(InterpolationSampling);
ENUM_DECLARE(ShaderStage);
ENUM_DECLARE(SeverityControl);
ENUM_DECLARE(Builtin);
ENUM_DECLARE(Extension);
ENUM_DECLARE(LanguageFeature);

#undef ENUM_DECLARE

AccessMode defaultAccessModeForAddressSpace(AddressSpace);

} // namespace WGSL

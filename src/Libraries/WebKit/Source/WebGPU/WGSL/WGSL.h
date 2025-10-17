/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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

#include "CompilationMessage.h"
#include "CompilationScope.h"
#include "ConstantValue.h"
#include "WGSLEnums.h"
#include <cinttypes>
#include <cstdint>
#include <memory>
#include <variant>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/OptionSet.h>
#include <wtf/UniqueRef.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WGSL {

//
// Step 1
//

class ShaderModule;
class CompilationScope;

namespace AST {
class Expression;
}

struct SuccessfulCheck {
    SuccessfulCheck() = delete;
    SuccessfulCheck(SuccessfulCheck&&);
    SuccessfulCheck(Vector<Warning>&&, UniqueRef<ShaderModule>&&);
    ~SuccessfulCheck();
    Vector<Warning> warnings;
    UniqueRef<ShaderModule> ast;
};

struct FailedCheck {
    Vector<Error> errors;
    Vector<Warning> warnings;
};

struct SourceMap {
    // I don't know what goes in here.
    // https://sourcemaps.info/spec.html
};

struct Configuration {
    uint32_t maxBuffersPlusVertexBuffersForVertexStage = 8;
    uint32_t maxBuffersForFragmentStage = 8;
    uint32_t maxBuffersForComputeStage = 8;
    uint32_t maximumCombinedWorkgroupVariablesSize = 16384;
    const HashSet<String> supportedFeatures = { };
};

std::variant<SuccessfulCheck, FailedCheck> staticCheck(const String& wgsl, const std::optional<SourceMap>&, const Configuration&);

//
// Step 2
//

enum class BufferBindingType : uint8_t {
    Uniform,
    Storage,
    ReadOnlyStorage
};

struct BufferBindingLayout {
    BufferBindingType type;
    bool hasDynamicOffset;
    uint64_t minBindingSize;
};

enum class SamplerBindingType : uint8_t {
    Filtering,
    NonFiltering,
    Comparison
};

struct SamplerBindingLayout {
    SamplerBindingType type;
};

enum class TextureSampleType : uint8_t {
    Float,
    UnfilterableFloat,
    Depth,
    SignedInt,
    UnsignedInt
};

enum class TextureViewDimension : uint8_t {
    OneDimensional,
    TwoDimensional,
    TwoDimensionalArray,
    Cube,
    CubeArray,
    ThreeDimensional
};

struct TextureBindingLayout {
    TextureSampleType sampleType;
    TextureViewDimension viewDimension;
    bool multisampled;
};

enum class StorageTextureAccess : uint8_t {
    WriteOnly,
    ReadOnly,
    ReadWrite,
};

struct StorageTextureBindingLayout {
    StorageTextureAccess access { StorageTextureAccess::WriteOnly };
    TexelFormat format;
    TextureViewDimension viewDimension;
};

struct ExternalTextureBindingLayout {
    // Sentinel
};

struct BindGroupLayoutEntry {
    uint32_t binding;
    uint32_t webBinding;
    OptionSet<ShaderStage> visibility;
    using BindingMember = std::variant<BufferBindingLayout, SamplerBindingLayout, TextureBindingLayout, StorageTextureBindingLayout, ExternalTextureBindingLayout>;
    BindingMember bindingMember;
    String name;
    std::optional<uint32_t> vertexArgumentBufferIndex;
    std::optional<uint32_t> vertexArgumentBufferSizeIndex;
    std::optional<uint32_t> vertexBufferDynamicOffset;

    std::optional<uint32_t> fragmentArgumentBufferIndex;
    std::optional<uint32_t> fragmentArgumentBufferSizeIndex;
    std::optional<uint32_t> fragmentBufferDynamicOffset;

    std::optional<uint32_t> computeArgumentBufferIndex;
    std::optional<uint32_t> computeArgumentBufferSizeIndex;
    std::optional<uint32_t> computeBufferDynamicOffset;
};

struct BindGroupLayout {
    // Metal's [[id(n)]] indices are equal to the index into this vector.
    uint32_t group;
    Vector<BindGroupLayoutEntry> entries;
};

struct PipelineLayout {
    // Metal's [[buffer(n)]] indices are equal to the index into this vector.
    Vector<BindGroupLayout> bindGroupLayouts;
};

namespace Reflection {

struct Vertex {
    bool invariantIsPresent;
    // Tons of reflection data to appear here in the future.
};

struct Fragment {
    // Tons of reflection data to appear here in the future.
};

struct WorkgroupSize {
    const AST::Expression* width;
    const AST::Expression* height;
    const AST::Expression* depth;
};

struct Compute {
    WorkgroupSize workgroupSize;
};

enum class SpecializationConstantType : uint8_t {
    Boolean,
    Float,
    Int,
    Unsigned,
    Half
};

struct SpecializationConstant {
    String mangledName;
    SpecializationConstantType type;
    AST::Expression* defaultValue;
};

struct EntryPointInformation {
    // FIXME: This can probably be factored better.
    String originalName;
    String mangledName;
    std::optional<PipelineLayout> defaultLayout; // If the input PipelineLayout is nullopt, the compiler computes a layout and returns it. https://gpuweb.github.io/gpuweb/#default-pipeline-layout
    HashMap<String, SpecializationConstant> specializationConstants;
    std::variant<Vertex, Fragment, Compute> typedEntryPoint;
    size_t sizeForWorkgroupVariables { 0 };
};

} // namespace Reflection

struct PrepareResult {
    HashMap<String, Reflection::EntryPointInformation> entryPoints;
    CompilationScope compilationScope;
};

std::variant<PrepareResult, Error> prepare(ShaderModule&, const HashMap<String, PipelineLayout*>&);
std::variant<PrepareResult, Error> prepare(ShaderModule&, const String& entryPointName, PipelineLayout*);

std::variant<String, Error> generate(ShaderModule&, PrepareResult&, HashMap<String, ConstantValue>&);

std::optional<ConstantValue> evaluate(const AST::Expression&, const HashMap<String, ConstantValue>&);

} // namespace WGSL

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 19, 2022.
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

#import "ASTInterpolateAttribute.h"
#import "WGSL.h"
#import <wtf/FastMalloc.h>
#import <wtf/Ref.h>
#import <wtf/RefCounted.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/text/StringHash.h>
#import <wtf/text/WTFString.h>

struct WGPUShaderModuleImpl {
};

namespace WGSL {
namespace AST {
class Function;
}
struct Type;
}

namespace WebGPU {

class Device;
class PipelineLayout;

// https://gpuweb.github.io/gpuweb/#gpushadermodule
class ShaderModule : public WGPUShaderModuleImpl, public RefCounted<ShaderModule> {
    WTF_MAKE_TZONE_ALLOCATED(ShaderModule);

    using CheckResult = std::variant<WGSL::SuccessfulCheck, WGSL::FailedCheck, std::monostate>;
public:
    static Ref<ShaderModule> create(std::variant<WGSL::SuccessfulCheck, WGSL::FailedCheck>&& checkResult, HashMap<String, Ref<PipelineLayout>>&& pipelineLayoutHints, HashMap<String, WGSL::Reflection::EntryPointInformation>&& entryPointInformation, id<MTLLibrary> library, Device& device)
    {
        return adoptRef(*new ShaderModule(WTFMove(checkResult), WTFMove(pipelineLayoutHints), WTFMove(entryPointInformation), library, device));
    }
    static Ref<ShaderModule> createInvalid(Device& device, CheckResult&& checkResult = std::monostate { })
    {
        return adoptRef(*new ShaderModule(device, WTFMove(checkResult)));
    }

    ~ShaderModule();

    void getCompilationInfo(CompletionHandler<void(WGPUCompilationInfoRequestStatus, const WGPUCompilationInfo&)>&& callback);
    void setLabel(String&&);

    bool isValid() const { return std::holds_alternative<WGSL::SuccessfulCheck>(m_checkResult); }

    static WGSL::PipelineLayout convertPipelineLayout(const PipelineLayout&);
    static id<MTLLibrary> createLibrary(id<MTLDevice>, const String& msl, String&& label, NSError **);

    WGSL::ShaderModule* ast() const;

    const PipelineLayout* pipelineLayoutHint(const String&) const;
    const WGSL::Reflection::EntryPointInformation* entryPointInformation(const String&) const;
    id<MTLLibrary> library() const { return m_library; }

    Device& device() const { return m_device; }
    const String& defaultVertexEntryPoint() const;
    const String& defaultFragmentEntryPoint() const;
    const String& defaultComputeEntryPoint() const;

    using VertexStageIn = HashMap<uint32_t, WGPUVertexFormat, DefaultHash<uint32_t>, WTF::UnsignedWithZeroKeyHashTraits<uint32_t>>;
    using FragmentOutputs = HashMap<uint32_t, MTLDataType, DefaultHash<uint32_t>, WTF::UnsignedWithZeroKeyHashTraits<uint32_t>>;
    struct VertexOutputFragmentInput {
        MTLDataType dataType { MTLDataTypeNone };
        std::optional<WGSL::AST::Interpolation> interpolation { std::nullopt };
    };
    using VertexOutputs = HashMap<uint32_t, VertexOutputFragmentInput, DefaultHash<uint32_t>, WTF::UnsignedWithZeroKeyHashTraits<uint32_t>>;
    using FragmentInputs = VertexOutputs;
    const FragmentOutputs* fragmentReturnTypeForEntryPoint(const String&) const;
    const FragmentInputs* fragmentInputsForEntryPoint(const String&) const;
    const VertexStageIn* stageInTypesForEntryPoint(const String&) const;
    const VertexOutputs* vertexReturnTypeForEntryPoint(const String&) const;
    bool usesFrontFacingInInput(const String&) const;
    bool usesSampleIndexInInput(const String&) const;
    bool usesSampleMaskInInput(const String&) const;
    bool usesSampleMaskInOutput(const String&) const;
    bool usesFragDepth(const String&) const;

private:
    ShaderModule(std::variant<WGSL::SuccessfulCheck, WGSL::FailedCheck>&&, HashMap<String, Ref<PipelineLayout>>&&, HashMap<String, WGSL::Reflection::EntryPointInformation>&&, id<MTLLibrary>, Device&);
    ShaderModule(Device&, CheckResult&&);

    CheckResult convertCheckResult(std::variant<WGSL::SuccessfulCheck, WGSL::FailedCheck>&&);

    const CheckResult m_checkResult;
    const HashMap<String, Ref<PipelineLayout>> m_pipelineLayoutHints;
    const HashMap<String, WGSL::Reflection::EntryPointInformation> m_entryPointInformation;
    const id<MTLLibrary> m_library { nil }; // This is only non-null if we could compile the module early.
    void populateFragmentInputs(const WGSL::Type&, ShaderModule::FragmentInputs&, const String&);
    FragmentInputs parseFragmentInputs(const WGSL::AST::Function&);
    void populateOutputState(const String&, WGSL::Builtin);

    ShaderModule::FragmentOutputs parseFragmentReturnType(const WGSL::Type&, const String&);

    const Ref<Device> m_device;
    // FIXME: https://bugs.webkit.org/show_bug.cgi?id=250441 - this needs to be populated from the compiler
    HashMap<String, String> m_constantIdentifiersToNames;
    HashMap<String, FragmentOutputs> m_fragmentReturnTypeForEntryPoint;
    HashMap<String, FragmentInputs> m_fragmentInputsForEntryPoint;
    HashMap<String, VertexOutputs> m_vertexReturnTypeForEntryPoint;
    HashMap<String, VertexStageIn> m_stageInTypesForEntryPoint;

    String m_defaultVertexEntryPoint;
    String m_defaultFragmentEntryPoint;
    String m_defaultComputeEntryPoint;

    struct ShaderModuleState {
        bool usesFrontFacingInInput { false };
        bool usesSampleIndexInInput { false };
        bool usesSampleMaskInInput { false };
        bool usesSampleMaskInOutput { false };
        bool usesFragDepth { false };
    };
    const ShaderModuleState* shaderModuleState(const String&) const;
    ShaderModuleState& populateShaderModuleState(const String&);
    HashMap<String, ShaderModuleState> m_usageInformationPerEntryPoint;
};

} // namespace WebGPU

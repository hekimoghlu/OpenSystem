/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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

#if ENABLE(FTL_JIT)

#include "DFGCommonData.h"
#include "FTLLazySlowPath.h"
#include "FTLOSRExit.h"
#include "JITCode.h"
#include "JITOpaqueByproducts.h"

namespace JSC {

class TrackedReferences;

namespace FTL {

class JITCode : public JSC::JITCode {
public:
    JITCode();
    ~JITCode() override;

    CodePtr<JSEntryPtrTag> addressForCall(ArityCheckMode) override;
    void* executableAddressAtOffset(size_t offset) override;
    void* dataAddressAtOffset(size_t offset) override;
    unsigned offsetOf(void* pointerIntoCode) override;
    size_t size() override;
    void setSize(size_t size) { m_size = size; }
    bool contains(void*) override;

    void initializeB3Code(CodeRef<JSEntryPtrTag>);
    void initializeB3Byproducts(std::unique_ptr<OpaqueByproducts>);
    void initializeAddressForCall(CodePtr<JSEntryPtrTag>);
    void initializeAddressForArityCheck(CodePtr<JSEntryPtrTag>);
    
    void validateReferences(const TrackedReferences&) override;

    RegisterSetBuilder liveRegistersToPreserveAtExceptionHandlingCallSite(CodeBlock*, CallSiteIndex) override;

    std::optional<CodeOrigin> findPC(CodeBlock*, void* pc) override;

    CodeRef<JSEntryPtrTag> b3Code() const { return m_b3Code; }
    
    JITCode* ftl() override;
    DFG::CommonData* dfgCommon() override;
    const DFG::CommonData* dfgCommon() const override;
    static constexpr ptrdiff_t commonDataOffset() { return OBJECT_OFFSETOF(JITCode, common); }
    void shrinkToFit() override;

    bool isUnlinked() const { return common.isUnlinked(); }

    PCToCodeOriginMap* pcToCodeOriginMap() override { return common.m_pcToCodeOriginMap.get(); }

    const RegisterAtOffsetList* calleeSaveRegisters() const { return &m_calleeSaveRegisters; }

    unsigned numberOfCompiledDFGNodes() const { return m_numberOfCompiledDFGNodes; }
    void setNumberOfCompiledDFGNodes(unsigned numberOfCompiledDFGNodes)
    {
        m_numberOfCompiledDFGNodes = numberOfCompiledDFGNodes;
    }
    
    DFG::CommonData common;
    Vector<OSRExit> m_osrExit;
    RegisterAtOffsetList m_calleeSaveRegisters;
    SegmentedVector<OSRExitDescriptor, 8> osrExitDescriptors;
    Vector<std::unique_ptr<LazySlowPath>> lazySlowPaths;
    
private:
    CodeRef<JSEntryPtrTag> m_b3Code;
    std::unique_ptr<OpaqueByproducts> m_b3Byproducts;
    CodePtr<JSEntryPtrTag> m_addressForArityCheck;
    size_t m_size { 1000 };
    unsigned m_numberOfCompiledDFGNodes { 0 };
};

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 30, 2025.
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
#include "config.h"
#include "FTLJITCode.h"

#if ENABLE(FTL_JIT)

#include "FTLState.h"
#include "JSCPtrTag.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC { namespace FTL {

using namespace B3;

JITCode::JITCode()
    : JSC::JITCode(JITType::FTLJIT)
    , common(/* isUnlinked */ false)
{
}

JITCode::~JITCode()
{
    if (FTL::shouldDumpDisassembly()) {
        if (m_b3Code)
            dataLogLn("Destroying FTL JIT code at ", m_b3Code);
    }
}

void JITCode::initializeB3Code(CodeRef<JSEntryPtrTag> b3Code)
{
    m_b3Code = b3Code;
}

void JITCode::initializeB3Byproducts(std::unique_ptr<OpaqueByproducts> byproducts)
{
    m_b3Byproducts = WTFMove(byproducts);
}

void JITCode::initializeAddressForCall(CodePtr<JSEntryPtrTag> address)
{
    m_addressForCall = address;
}

void JITCode::initializeAddressForArityCheck(CodePtr<JSEntryPtrTag> entrypoint)
{
    m_addressForArityCheck = entrypoint;
}

CodePtr<JSEntryPtrTag> JITCode::addressForCall(ArityCheckMode arityCheck)
{
    switch (arityCheck) {
    case ArityCheckNotRequired:
        return m_addressForCall;
    case MustCheckArity:
        return m_addressForArityCheck;
    }
    RELEASE_ASSERT_NOT_REACHED();
    return CodePtr<JSEntryPtrTag>();
}

void* JITCode::executableAddressAtOffset(size_t offset)
{
    if (!offset)
        return m_addressForCall.taggedPtr();

    char* executableAddress = m_addressForCall.untaggedPtr<char*>();
    return tagCodePtr<JSEntryPtrTag>(executableAddress + offset);
}

void* JITCode::dataAddressAtOffset(size_t)
{
    // We can't patch FTL code, yet. Even if we did, it's not clear that we would do so
    // through this API.
    RELEASE_ASSERT_NOT_REACHED();
    return nullptr;
}

unsigned JITCode::offsetOf(void*)
{
    // We currently don't have visibility into the FTL code.
    RELEASE_ASSERT_NOT_REACHED();
    return 0;
}

size_t JITCode::size()
{
    // We don't know the size of FTL code, yet. Make a wild guess. This is mostly used for
    // GC load estimates.
    return m_size;
}

bool JITCode::contains(void*)
{
    // We have no idea what addresses the FTL code contains, yet.
    RELEASE_ASSERT_NOT_REACHED();
    return false;
}

JITCode* JITCode::ftl()
{
    return this;
}

DFG::CommonData* JITCode::dfgCommon()
{
    return &common;
}

const DFG::CommonData* JITCode::dfgCommon() const
{
    return &common;
}

void JITCode::shrinkToFit()
{
    common.shrinkToFit();
    m_osrExit.shrinkToFit();
    osrExitDescriptors.shrinkToFit();
    lazySlowPaths.shrinkToFit();
}

void JITCode::validateReferences(const TrackedReferences& trackedReferences)
{
    common.validateReferences(trackedReferences);
    
    for (OSRExit& exit : m_osrExit)
        exit.m_descriptor->validateReferences(trackedReferences);
}

RegisterSetBuilder JITCode::liveRegistersToPreserveAtExceptionHandlingCallSite(CodeBlock*, CallSiteIndex callSiteIndex)
{
    for (OSRExit& exit : m_osrExit) {
        if (exit.m_exceptionHandlerCallSiteIndex.bits() == callSiteIndex.bits()) {
            RELEASE_ASSERT(exit.isExceptionHandler());
            RELEASE_ASSERT(exit.isGenericUnwindHandler());
            return ValueRep::usedRegisters(/* isSIMDContext = */ false, exit.m_valueReps);
        }
    }
    return { };
}

std::optional<CodeOrigin> JITCode::findPC(CodeBlock* codeBlock, void* pc)
{
    for (OSRExit& exit : m_osrExit) {
        if (ExecutableMemoryHandle* handle = exit.m_code.executableMemory()) {
            if (handle->start().untaggedPtr() <= pc && pc < handle->end().untaggedPtr())
                return std::optional<CodeOrigin>(exit.m_codeOriginForExitProfile);
        }
    }

    for (std::unique_ptr<LazySlowPath>& lazySlowPath : lazySlowPaths) {
        if (ExecutableMemoryHandle* handle = lazySlowPath->stub().executableMemory()) {
            if (handle->start().untaggedPtr() <= pc && pc < handle->end().untaggedPtr())
                return std::optional<CodeOrigin>(codeBlock->codeOrigin(lazySlowPath->callSiteIndex()));
        }
    }

    return std::nullopt;
}

} } // namespace JSC::FTL

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(FTL_JIT)

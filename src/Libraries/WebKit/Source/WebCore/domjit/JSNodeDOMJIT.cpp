/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 18, 2024.
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
#include "JSNode.h"

#if ENABLE(JIT)

#include "DOMJITAbstractHeapRepository.h"
#include "DOMJITCheckDOM.h"
#include "DOMJITHelpers.h"
#include "JSDOMWrapper.h"
#include "Node.h"
#include <JavaScriptCore/FrameTracers.h>
#include <JavaScriptCore/Snippet.h>
#include <JavaScriptCore/SnippetParams.h>

namespace WebCore {
using namespace JSC;

Ref<JSC::Snippet> checkSubClassSnippetForJSNode()
{
    return DOMJIT::checkDOM<Node>();
}

enum class IsContainerGuardRequirement { Required, NotRequired };

template<typename WrappedNode, typename ToWrapperOperation>
static Ref<JSC::DOMJIT::CallDOMGetterSnippet> createCallDOMGetterForOffsetAccess(ptrdiff_t offset, ToWrapperOperation operation, IsContainerGuardRequirement isContainerGuardRequirement, std::optional<uintptr_t> mask = std::nullopt)
{
    Ref<JSC::DOMJIT::CallDOMGetterSnippet> snippet = JSC::DOMJIT::CallDOMGetterSnippet::create();
    snippet->numGPScratchRegisters = 1;
    snippet->setGenerator([=](CCallHelpers& jit, JSC::SnippetParams& params) {
        JSValueRegs result = params[0].jsValueRegs();
        GPRReg node = params[1].gpr();
        GPRReg globalObject = params[2].gpr();
        GPRReg scratch = params.gpScratch(0);
        JSValue globalObjectValue = params[2].value();

        CCallHelpers::JumpList nullCases;
        // Load a wrapped object. "node" should be already type checked by CheckDOM.
        jit.loadPtr(CCallHelpers::Address(node, JSNode::offsetOfWrapped()), scratch);
        static_assert(!JSNode::hasCustomPtrTraits(), "Optimized JSNode wrapper access should not be using RawPtrTraits");

        if (isContainerGuardRequirement == IsContainerGuardRequirement::Required)
            nullCases.append(jit.branchTest16(CCallHelpers::Zero, CCallHelpers::Address(scratch, Node::typeFlagsMemoryOffset()), CCallHelpers::TrustedImm32(Node::flagIsContainer())));

        jit.loadPtr(CCallHelpers::Address(scratch, offset), scratch);
        if (mask)
            jit.andPtr(CCallHelpers::TrustedImmPtr(*mask), scratch);
        nullCases.append(jit.branchTestPtr(CCallHelpers::Zero, scratch));

        DOMJIT::toWrapper<WrappedNode>(jit, params, scratch, globalObject, result, operation, globalObjectValue);
        CCallHelpers::Jump done = jit.jump();

        nullCases.link(&jit);
        jit.moveValue(jsNull(), result);
        done.link(&jit);
        return CCallHelpers::JumpList();
    });
    return snippet;
}

Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileNodeFirstChildAttribute()
{
    auto snippet = createCallDOMGetterForOffsetAccess<Node>(CAST_OFFSET(Node*, ContainerNode*) + ContainerNode::firstChildMemoryOffset(), DOMJIT::operationToJSNode, IsContainerGuardRequirement::Required);
    snippet->effect = JSC::DOMJIT::Effect::forDef(DOMJIT::AbstractHeapRepository::Node_firstChild);
    return snippet;
}

Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileNodeLastChildAttribute()
{
    auto snippet = createCallDOMGetterForOffsetAccess<Node>(CAST_OFFSET(Node*, ContainerNode*) + ContainerNode::lastChildMemoryOffset(), DOMJIT::operationToJSNode, IsContainerGuardRequirement::Required);
    snippet->effect = JSC::DOMJIT::Effect::forDef(DOMJIT::AbstractHeapRepository::Node_lastChild);
    return snippet;
}

Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileNodeNextSiblingAttribute()
{
    auto snippet = createCallDOMGetterForOffsetAccess<Node>(Node::nextSiblingMemoryOffset(), DOMJIT::operationToJSNode, IsContainerGuardRequirement::NotRequired);
    snippet->effect = JSC::DOMJIT::Effect::forDef(DOMJIT::AbstractHeapRepository::Node_nextSibling);
    return snippet;
}

Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileNodePreviousSiblingAttribute()
{
    auto snippet = createCallDOMGetterForOffsetAccess<Node>(Node::previousSiblingMemoryOffset(), DOMJIT::operationToJSNode, IsContainerGuardRequirement::NotRequired);
    snippet->effect = JSC::DOMJIT::Effect::forDef(DOMJIT::AbstractHeapRepository::Node_previousSibling);
    return snippet;
}

Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileNodeParentNodeAttribute()
{
    auto snippet = createCallDOMGetterForOffsetAccess<ContainerNode>(Node::parentNodeMemoryOffset(), DOMJIT::operationToJSContainerNode, IsContainerGuardRequirement::NotRequired);
    snippet->effect = JSC::DOMJIT::Effect::forDef(DOMJIT::AbstractHeapRepository::Node_parentNode);
    return snippet;
}

Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileNodeNodeTypeAttribute()
{
    Ref<JSC::DOMJIT::CallDOMGetterSnippet> snippet = JSC::DOMJIT::CallDOMGetterSnippet::create();
    snippet->effect = JSC::DOMJIT::Effect::forPure();
    snippet->requireGlobalObject = false;
    snippet->setGenerator([=](CCallHelpers& jit, JSC::SnippetParams& params) {
        JSValueRegs result = params[0].jsValueRegs();
        GPRReg node = params[1].gpr();
        jit.load8(CCallHelpers::Address(node, JSC::JSCell::typeInfoTypeOffset()), result.payloadGPR());
        jit.and32(CCallHelpers::TrustedImm32(JSNodeTypeMask), result.payloadGPR());
        jit.boxInt32(result.payloadGPR(), result);
        return CCallHelpers::JumpList();
    });
    return snippet;
}

Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileNodeOwnerDocumentAttribute()
{
    Ref<JSC::DOMJIT::CallDOMGetterSnippet> snippet = JSC::DOMJIT::CallDOMGetterSnippet::create();
    snippet->numGPScratchRegisters = 2;
    snippet->setGenerator([=](CCallHelpers& jit, JSC::SnippetParams& params) {
        JSValueRegs result = params[0].jsValueRegs();
        GPRReg node = params[1].gpr();
        GPRReg globalObject = params[2].gpr();
        JSValue globalObjectValue = params[2].value();
        GPRReg wrapped = params.gpScratch(0);
        GPRReg document = params.gpScratch(1);

        jit.loadPtr(CCallHelpers::Address(node, JSNode::offsetOfWrapped()), wrapped);
        static_assert(!JSNode::hasCustomPtrTraits(), "Optimized JSNode wrapper access should not be using RawPtrTraits");
        
        DOMJIT::loadDocument(jit, wrapped, document);
        RELEASE_ASSERT(!CAST_OFFSET(EventTarget*, Node*));
        RELEASE_ASSERT(!CAST_OFFSET(Node*, Document*));

        CCallHelpers::JumpList nullCases;
        // If the |this| is the document itself, ownerDocument will return null.
        nullCases.append(jit.branchPtr(CCallHelpers::Equal, wrapped, document));
        DOMJIT::toWrapper<Document>(jit, params, document, globalObject, result, DOMJIT::operationToJSDocument, globalObjectValue);
        auto done = jit.jump();

        nullCases.link(&jit);
        jit.moveValue(jsNull(), result);
        done.link(&jit);
        return CCallHelpers::JumpList();
    });
    snippet->effect = JSC::DOMJIT::Effect::forDef(DOMJIT::AbstractHeapRepository::Node_ownerDocument);
    return snippet;
}

}

#endif

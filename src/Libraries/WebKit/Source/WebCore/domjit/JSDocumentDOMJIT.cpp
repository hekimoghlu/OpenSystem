/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 11, 2024.
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
#include "JSDocument.h"

#if ENABLE(JIT)

#include "DOMJITAbstractHeapRepository.h"
#include "DOMJITCheckDOM.h"
#include "DOMJITHelpers.h"
#include "Document.h"
#include "JSDOMWrapper.h"
#include "JSElement.h"
#include "JSHTMLElement.h"
#include <JavaScriptCore/Snippet.h>
#include <JavaScriptCore/SnippetParams.h>

IGNORE_WARNINGS_BEGIN("frame-address");

namespace WebCore {
using namespace JSC;

Ref<JSC::Snippet> checkSubClassSnippetForJSDocument()
{
    return DOMJIT::checkDOM<Document>();
}

Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileDocumentDocumentElementAttribute()
{
    Ref<JSC::DOMJIT::CallDOMGetterSnippet> snippet = JSC::DOMJIT::CallDOMGetterSnippet::create();
    snippet->numGPScratchRegisters = 1;
    snippet->setGenerator([=](CCallHelpers& jit, JSC::SnippetParams& params) {
        JSValueRegs result = params[0].jsValueRegs();
        GPRReg document = params[1].gpr();
        GPRReg globalObject = params[2].gpr();
        JSValue globalObjectValue = params[2].value();
        GPRReg scratch = params.gpScratch(0);

        jit.loadPtr(CCallHelpers::Address(document, JSDocument::offsetOfWrapped()), scratch);
        static_assert(!JSDocument::hasCustomPtrTraits(), "Optimized JSDocument wrapper access should not be using RawPtrTraits");

        DOMJIT::loadDocumentElement(jit, scratch, scratch);
        auto nullCase = jit.branchTestPtr(CCallHelpers::Zero, scratch);
        DOMJIT::toWrapper<Element>(jit, params, scratch, globalObject, result, DOMJIT::operationToJSElement, globalObjectValue);
        auto done = jit.jump();

        nullCase.link(&jit);
        jit.moveValue(jsNull(), result);
        done.link(&jit);

        return CCallHelpers::JumpList();
    });
    snippet->effect = JSC::DOMJIT::Effect::forDef(DOMJIT::AbstractHeapRepository::Document_documentElement);
    return snippet;
}

static void loadLocalName(CCallHelpers& jit, GPRReg htmlElement, GPRReg localNameImpl)
{
    jit.loadPtr(CCallHelpers::Address(htmlElement, Element::tagQNameMemoryOffset() + QualifiedName::implMemoryOffset()), localNameImpl);
    jit.loadPtr(CCallHelpers::Address(localNameImpl, QualifiedName::QualifiedNameImpl::localNameMemoryOffset()), localNameImpl);
}

Ref<JSC::DOMJIT::CallDOMGetterSnippet> compileDocumentBodyAttribute()
{
    Ref<JSC::DOMJIT::CallDOMGetterSnippet> snippet = JSC::DOMJIT::CallDOMGetterSnippet::create();
    snippet->numGPScratchRegisters = 2;
    snippet->setGenerator([=](CCallHelpers& jit, JSC::SnippetParams& params) {
        JSValueRegs result = params[0].jsValueRegs();
        GPRReg document = params[1].gpr();
        GPRReg globalObject = params[2].gpr();
        JSValue globalObjectValue = params[2].value();
        GPRReg scratch1 = params.gpScratch(0);
        GPRReg scratch2 = params.gpScratch(1);

        jit.loadPtr(CCallHelpers::Address(document, JSDocument::offsetOfWrapped()), scratch1);
        static_assert(!JSDocument::hasCustomPtrTraits(), "Optimized JSDocument wrapper access should not be using RawPtrTraits");

        DOMJIT::loadDocumentElement(jit, scratch1, scratch1);

        CCallHelpers::JumpList nullCases;
        CCallHelpers::JumpList successCases;
        nullCases.append(jit.branchTestPtr(CCallHelpers::Zero, scratch1));
        nullCases.append(DOMJIT::branchTestIsHTMLFlagOnNode(jit, CCallHelpers::Zero, scratch1));
        // We ensured that the name of the given element is HTML qualified.
        // It allows us to perform local name comparison!
        loadLocalName(jit, scratch1, scratch2);
        nullCases.append(jit.branchPtr(CCallHelpers::NotEqual, scratch2, CCallHelpers::TrustedImmPtr(HTMLNames::htmlTag->localName().impl())));

        RELEASE_ASSERT(!CAST_OFFSET(Node*, ContainerNode*));
        RELEASE_ASSERT(!CAST_OFFSET(Node*, Element*));
        RELEASE_ASSERT(!CAST_OFFSET(Node*, HTMLElement*));

        // Node* node = current.firstChild();
        // while (node && !is<HTMLElement>(*node))
        //     node = node->nextSibling();
        // return downcast<HTMLElement>(node);
        jit.loadPtr(CCallHelpers::Address(scratch1, ContainerNode::firstChildMemoryOffset()), scratch1);

        CCallHelpers::Label loopStart = jit.label();
        nullCases.append(jit.branchTestPtr(CCallHelpers::Zero, scratch1));
        auto notHTMLElementCase = DOMJIT::branchTestIsHTMLFlagOnNode(jit, CCallHelpers::Zero, scratch1);
        // We ensured that the name of the given element is HTML qualified.
        // It allows us to perform local name comparison!
        loadLocalName(jit, scratch1, scratch2);
        successCases.append(jit.branchPtr(CCallHelpers::Equal, scratch2, CCallHelpers::TrustedImmPtr(HTMLNames::bodyTag->localName().impl())));
        successCases.append(jit.branchPtr(CCallHelpers::Equal, scratch2, CCallHelpers::TrustedImmPtr(HTMLNames::framesetTag->localName().impl())));

        notHTMLElementCase.link(&jit);
        jit.loadPtr(CCallHelpers::Address(scratch1, Node::nextSiblingMemoryOffset()), scratch1);
        jit.jump().linkTo(loopStart, &jit);

        successCases.link(&jit);
        DOMJIT::toWrapper<HTMLElement>(jit, params, scratch1, globalObject, result, DOMJIT::operationToJSHTMLElement, globalObjectValue);
        auto done = jit.jump();

        nullCases.link(&jit);
        jit.moveValue(jsNull(), result);
        done.link(&jit);

        return CCallHelpers::JumpList();
    });
    snippet->effect = JSC::DOMJIT::Effect::forDef(DOMJIT::AbstractHeapRepository::Document_body);
    return snippet;
}

namespace DOMJIT {

JSC_DEFINE_JIT_OPERATION(operationToJSElement, JSC::EncodedJSValue, (JSC::JSGlobalObject* globalObject, void* result))
{
    ASSERT(result);
    ASSERT(globalObject);
    JSC::VM& vm = globalObject->vm();
    JSC::CallFrame* callFrame = DECLARE_CALL_FRAME(vm);
    JSC::JITOperationPrologueCallFrameTracer tracer(vm, callFrame);
    auto scope = DECLARE_THROW_SCOPE(vm);
    OPERATION_RETURN(scope, DOMJIT::toWrapperSlowImpl<Element>(globalObject, result));
}

JSC_DEFINE_JIT_OPERATION(operationToJSHTMLElement, JSC::EncodedJSValue, (JSC::JSGlobalObject* globalObject, void* result))
{
    ASSERT(result);
    ASSERT(globalObject);
    JSC::VM& vm = globalObject->vm();
    JSC::CallFrame* callFrame = DECLARE_CALL_FRAME(vm);
    JSC::JITOperationPrologueCallFrameTracer tracer(vm, callFrame);
    auto scope = DECLARE_THROW_SCOPE(vm);
    OPERATION_RETURN(scope, DOMJIT::toWrapperSlowImpl<HTMLElement>(globalObject, result));
}

JSC_DEFINE_JIT_OPERATION(operationToJSDocument, JSC::EncodedJSValue, (JSC::JSGlobalObject* globalObject, void* result))
{
    ASSERT(result);
    ASSERT(globalObject);
    JSC::VM& vm = globalObject->vm();
    JSC::CallFrame* callFrame = DECLARE_CALL_FRAME(vm);
    JSC::JITOperationPrologueCallFrameTracer tracer(vm, callFrame);
    auto scope = DECLARE_THROW_SCOPE(vm);
    OPERATION_RETURN(scope, DOMJIT::toWrapperSlowImpl<Document>(globalObject, result));
}

JSC_DEFINE_JIT_OPERATION(operationToJSNode, JSC::EncodedJSValue, (JSC::JSGlobalObject* globalObject, void* result))
{
    ASSERT(result);
    ASSERT(globalObject);
    JSC::VM& vm = globalObject->vm();
    JSC::CallFrame* callFrame = DECLARE_CALL_FRAME(vm);
    JSC::JITOperationPrologueCallFrameTracer tracer(vm, callFrame);
    auto scope = DECLARE_THROW_SCOPE(vm);
    OPERATION_RETURN(scope, DOMJIT::toWrapperSlowImpl<Node>(globalObject, result));
}

JSC_DEFINE_JIT_OPERATION(operationToJSContainerNode, JSC::EncodedJSValue, (JSC::JSGlobalObject* globalObject, void* result))
{
    ASSERT(result);
    ASSERT(globalObject);
    JSC::VM& vm = globalObject->vm();
    JSC::CallFrame* callFrame = DECLARE_CALL_FRAME(vm);
    JSC::JITOperationPrologueCallFrameTracer tracer(vm, callFrame);
    auto scope = DECLARE_THROW_SCOPE(vm);
    OPERATION_RETURN(scope, DOMJIT::toWrapperSlowImpl<ContainerNode>(globalObject, result));
}

} } // namespace WebCore::DOMJIT

IGNORE_WARNINGS_END

#endif // ENABLE(JIT)

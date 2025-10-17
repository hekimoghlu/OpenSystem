/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 14, 2023.
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

#include "JSDOMGlobalObject.h"
#include "NodeConstants.h"
#include <JavaScriptCore/JSDestructibleObject.h>
#include <JavaScriptCore/StructureInlines.h>
#include <wtf/Compiler.h>
#include <wtf/SignedPtr.h>

namespace WebCore {

class ScriptExecutionContext;

// JSC allows us to extend JSType. If the highest 3 bits are set, we can add any Object types and they are
// recognized as OtherObj in JSC. And we encode Node type into JSType if the given JSType is subclass of Node.
// offset | 7 | 6 | 5 | 4   3   2   1   0  |
// value  | 1 | 1 | 1 | Non-node DOM types |
// If the given JSType is a subclass of Node, the format is the following.
// offset | 7 | 6 | 5 | 4 | 3   2   1   0  |
// value  | 1 | 1 | 1 | 1 |    NodeType    |

static const uint8_t JSDOMWrapperType                = 0b11101110;
static const uint8_t JSEventType                     = 0b11101111;
static const uint8_t JSNodeType                      = 0b11110000;
static const uint8_t JSNodeTypeMask                  = 0b00001111;
static const uint8_t JSTextNodeType                  = JSNodeType | NodeConstants::TEXT_NODE;
static const uint8_t JSProcessingInstructionNodeType = JSNodeType | NodeConstants::PROCESSING_INSTRUCTION_NODE;
static const uint8_t JSDocumentTypeNodeType          = JSNodeType | NodeConstants::DOCUMENT_TYPE_NODE;
static const uint8_t JSDocumentFragmentNodeType      = JSNodeType | NodeConstants::DOCUMENT_FRAGMENT_NODE;
static const uint8_t JSDocumentWrapperType           = JSNodeType | NodeConstants::DOCUMENT_NODE;
static const uint8_t JSCommentNodeType               = JSNodeType | NodeConstants::COMMENT_NODE;
static const uint8_t JSCDATASectionNodeType          = JSNodeType | NodeConstants::CDATA_SECTION_NODE;
static const uint8_t JSAttrNodeType                  = JSNodeType | NodeConstants::ATTRIBUTE_NODE;
static const uint8_t JSElementType                   = 0b11110000 | NodeConstants::ELEMENT_NODE;

static_assert(JSDOMWrapperType > JSC::LastJSCObjectType, "JSC::JSType offers the highest bit.");
static_assert(NodeConstants::LastNodeType <= JSNodeTypeMask, "NodeType should be represented in 4bit.");

class JSDOMObject : public JSC::JSDestructibleObject {
public:
    typedef JSC::JSDestructibleObject Base;

    template<typename, JSC::SubspaceAccess>
    static void subspaceFor(JSC::VM&) { RELEASE_ASSERT_NOT_REACHED(); }

    JSDOMGlobalObject* globalObject() const { return JSC::jsCast<JSDOMGlobalObject*>(JSC::JSNonFinalObject::globalObject()); }
    ScriptExecutionContext* scriptExecutionContext() const { return globalObject()->scriptExecutionContext(); }

protected:
    WEBCORE_EXPORT JSDOMObject(JSC::Structure*, JSC::JSGlobalObject&);
};

template<typename ImplementationClass, typename PtrTraits = RawPtrTraits<ImplementationClass>>
class JSDOMWrapper : public JSDOMObject {
public:
    using Base = JSDOMObject;
    using DOMWrapped = ImplementationClass;

    ImplementationClass& wrapped() const { return m_wrapped; }
    Ref<ImplementationClass> protectedWrapped() const { return m_wrapped; }
    static constexpr ptrdiff_t offsetOfWrapped() { return OBJECT_OFFSETOF(JSDOMWrapper, m_wrapped); }
    constexpr static bool hasCustomPtrTraits() { return !std::is_same_v<PtrTraits, RawPtrTraits<ImplementationClass>>; };
    
protected:
    JSDOMWrapper(JSC::Structure* structure, JSC::JSGlobalObject& globalObject, Ref<ImplementationClass>&& impl)
        : Base(structure, globalObject)
        , m_wrapped(WTFMove(impl)) { }

private:
    Ref<ImplementationClass, PtrTraits> m_wrapped;
};

template<typename ImplementationClass> struct JSDOMWrapperConverterTraits;

JSC::JSValue cloneAcrossWorlds(JSC::JSGlobalObject&, const JSDOMObject& owner, JSC::JSValue);

} // namespace WebCore

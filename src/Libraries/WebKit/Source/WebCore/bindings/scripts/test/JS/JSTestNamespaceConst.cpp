/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 12, 2024.
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
#include "JSTestNamespaceConst.h"

#include "ActiveDOMObject.h"
#include "ExtendedDOMClientIsoSubspaces.h"
#include "ExtendedDOMIsoSubspaces.h"
#include "JSDOMBinding.h"
#include "JSDOMConstructorNotCallable.h"
#include "JSDOMExceptionHandling.h"
#include "JSDOMGlobalObjectInlines.h"
#include "JSDOMWrapperCache.h"
#include "TestNamespaceConst.h"
#include "WebCoreJSClientData.h"
#include <JavaScriptCore/JSCInlines.h>
#include <JavaScriptCore/JSDestructibleObjectHeapCellType.h>
#include <JavaScriptCore/ObjectPrototype.h>
#include <JavaScriptCore/SlotVisitorMacros.h>
#include <JavaScriptCore/SubspaceInlines.h>
#include <wtf/GetPtr.h>
#include <wtf/PointerPreparations.h>

namespace WebCore {
using namespace JSC;

using JSTestNamespaceConstDOMConstructor = JSDOMConstructorNotCallable<JSTestNamespaceConst>;

/* Hash table for constructor */

static const std::array<HashTableValue, 2> JSTestNamespaceConstConstructorTableValues {
    HashTableValue { "TEST_FLAG"_s, PropertyAttribute::ReadOnly | PropertyAttribute::DontDelete | PropertyAttribute::ConstantInteger, NoIntrinsic, { HashTableValue::ConstantType, false } },
    HashTableValue { "TEST_BIT_MASK"_s, PropertyAttribute::ReadOnly | PropertyAttribute::DontDelete | PropertyAttribute::ConstantInteger, NoIntrinsic, { HashTableValue::ConstantType, 0x0000fc00 } },
};

static_assert(TestNamespaceConst::TEST_FLAG == false, "TEST_FLAG in TestNamespaceConst does not match value from IDL");
static_assert(TestNamespaceConst::TEST_BIT_MASK == 0x0000fc00, "TEST_BIT_MASK in TestNamespaceConst does not match value from IDL");

template<> const ClassInfo JSTestNamespaceConstDOMConstructor::s_info = { "TestNamespaceConst"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSTestNamespaceConstDOMConstructor) };

template<> JSValue JSTestNamespaceConstDOMConstructor::prototypeForStructure(JSC::VM& vm, const JSDOMGlobalObject& globalObject)
{
    UNUSED_PARAM(vm);
    return globalObject.objectPrototype();
}

template<> void JSTestNamespaceConstDOMConstructor::initializeProperties(VM& vm, JSDOMGlobalObject& globalObject)
{
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
    reifyStaticProperties(vm, JSTestNamespaceConst::info(), JSTestNamespaceConstConstructorTableValues, *this);
    UNUSED_PARAM(globalObject);
}

const ClassInfo JSTestNamespaceConst::s_info = { "TestNamespaceConst"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSTestNamespaceConst) };

JSTestNamespaceConst::JSTestNamespaceConst(Structure* structure, JSDOMGlobalObject& globalObject)
    : JSDOMObject(structure, globalObject) { }

static_assert(!std::is_base_of<ActiveDOMObject, TestNamespaceConst>::value, "Interface is not marked as [ActiveDOMObject] even though implementation class subclasses ActiveDOMObject.");

JSValue JSTestNamespaceConst::getConstructor(VM& vm, const JSGlobalObject* globalObject)
{
    return getDOMConstructor<JSTestNamespaceConstDOMConstructor, DOMConstructorID::TestNamespaceConst>(vm, *jsCast<const JSDOMGlobalObject*>(globalObject));
}

void JSTestNamespaceConst::destroy(JSC::JSCell* cell)
{
    JSTestNamespaceConst* thisObject = static_cast<JSTestNamespaceConst*>(cell);
    thisObject->JSTestNamespaceConst::~JSTestNamespaceConst();
}

JSC::GCClient::IsoSubspace* JSTestNamespaceConst::subspaceForImpl(JSC::VM& vm)
{
    return WebCore::subspaceForImpl<JSTestNamespaceConst, UseCustomHeapCellType::No>(vm,
        [] (auto& spaces) { return spaces.m_clientSubspaceForTestNamespaceConst.get(); },
        [] (auto& spaces, auto&& space) { spaces.m_clientSubspaceForTestNamespaceConst = std::forward<decltype(space)>(space); },
        [] (auto& spaces) { return spaces.m_subspaceForTestNamespaceConst.get(); },
        [] (auto& spaces, auto&& space) { spaces.m_subspaceForTestNamespaceConst = std::forward<decltype(space)>(space); }
    );
}


}

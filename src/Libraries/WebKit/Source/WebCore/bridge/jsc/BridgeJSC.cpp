/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 3, 2023.
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
#include "BridgeJSC.h"

#include "JSDOMWindowBase.h"
#include "LocalDOMWindow.h"
#include "runtime_object.h"
#include "runtime_root.h"
#include <JavaScriptCore/JSCInlines.h>
#include <JavaScriptCore/JSLock.h>
#include <JavaScriptCore/ObjectPrototype.h>
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

namespace Bindings {

WTF_MAKE_TZONE_ALLOCATED_IMPL(Class);
WTF_MAKE_TZONE_ALLOCATED_IMPL(Field);
WTF_MAKE_TZONE_ALLOCATED_IMPL(Method);

Array::Array(RefPtr<RootObject>&& rootObject)
    : m_rootObject(WTFMove(rootObject))
{
    ASSERT(m_rootObject);
}

Array::~Array() = default;

Instance::Instance(RefPtr<RootObject>&& rootObject)
    : m_rootObject(WTFMove(rootObject))
{
    ASSERT(m_rootObject);
}

Instance::~Instance()
{
    ASSERT(!m_runtimeObject);
}

void Instance::begin()
{
    virtualBegin();
}

void Instance::end()
{
    virtualEnd();
}

JSObject* Instance::createRuntimeObject(JSGlobalObject* lexicalGlobalObject)
{
    ASSERT(m_rootObject);
    ASSERT(m_rootObject->isValid());
    if (RuntimeObject* existingObject = m_runtimeObject.get())
        return existingObject;

    JSLockHolder lock(lexicalGlobalObject);
    RuntimeObject* newObject = newRuntimeObject(lexicalGlobalObject);
    m_runtimeObject = JSC::Weak<RuntimeObject>(newObject);
    m_rootObject->addRuntimeObject(lexicalGlobalObject->vm(), newObject);
    return newObject;
}

RuntimeObject* Instance::newRuntimeObject(JSGlobalObject* lexicalGlobalObject)
{
    JSLockHolder lock(lexicalGlobalObject);

    // FIXME: deprecatedGetDOMStructure uses the prototype off of the wrong global object.
    return RuntimeObject::create(lexicalGlobalObject->vm(), WebCore::deprecatedGetDOMStructure<RuntimeObject>(lexicalGlobalObject), this);
}

void Instance::willInvalidateRuntimeObject()
{
    m_runtimeObject.clear();
}

RootObject* Instance::rootObject() const
{
    return m_rootObject && m_rootObject->isValid() ? m_rootObject.get() : 0;
}

} // namespace Bindings

} // namespace JSC

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 14, 2025.
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
#include "runtime_root.h"

#include "BridgeJSC.h"
#include "runtime_object.h"
#include <JavaScriptCore/JSGlobalObject.h>
#include <JavaScriptCore/StrongInlines.h>
#include <JavaScriptCore/Weak.h>
#include <JavaScriptCore/WeakInlines.h>
#include <wtf/HashSet.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/Ref.h>
#include <wtf/StdLibExtras.h>

namespace JSC { namespace Bindings {

// This code attempts to solve two problems: (1) plug-ins leaking references to 
// JS and the DOM; (2) plug-ins holding stale references to JS and the DOM. Previous 
// comments in this file claimed that problem #1 was an issue in Java, in particular, 
// because Java, allegedly, didn't always call finalize when collecting an object.

typedef UncheckedKeyHashSet<RootObject*> RootObjectSet;

static RootObjectSet& rootObjectSet()
{
    static NeverDestroyed<RootObjectSet> staticRootObjectSet;
    return staticRootObjectSet;
}

// FIXME:  These two functions are a potential performance problem.  We could 
// fix them by adding a JSObject to RootObject dictionary.

RootObject* findProtectingRootObject(JSObject* jsObject)
{
    RootObjectSet::const_iterator end = rootObjectSet().end();
    for (RootObjectSet::const_iterator it = rootObjectSet().begin(); it != end; ++it) {
        if ((*it)->gcIsProtected(jsObject))
            return *it;
    }
    return 0;
}

RootObject* findRootObject(JSGlobalObject* globalObject)
{
    RootObjectSet::const_iterator end = rootObjectSet().end();
    for (RootObjectSet::const_iterator it = rootObjectSet().begin(); it != end; ++it) {
        if ((*it)->globalObject() == globalObject)
            return *it;
    }
    return 0;
}

RootObject::InvalidationCallback::~InvalidationCallback() = default;

Ref<RootObject> RootObject::create(const void* nativeHandle, JSGlobalObject* globalObject)
{
    return adoptRef(*new RootObject(nativeHandle, globalObject));
}

RootObject::RootObject(const void* nativeHandle, JSGlobalObject* globalObject)
    : m_isValid(true)
    , m_nativeHandle(nativeHandle)
    , m_globalObject(globalObject->vm(), globalObject)
{
    ASSERT(globalObject);
    rootObjectSet().add(this);
}

RootObject::~RootObject()
{
    if (m_isValid)
        invalidate();
}

void RootObject::invalidate()
{
    if (!m_isValid)
        return;

    {
        // Get the objects from the keys; the values might be nulled.
        // Safe because finalized runtime objects are removed from m_runtimeObjects by RootObject::finalize.
        for (RuntimeObject* runtimeObject : m_runtimeObjects.keys())
            runtimeObject->invalidate();

        m_runtimeObjects.clear();
    }

    m_isValid = false;

    m_nativeHandle = 0;
    m_globalObject.clear();

    {
        UncheckedKeyHashSet<InvalidationCallback*>::iterator end = m_invalidationCallbacks.end();
        for (UncheckedKeyHashSet<InvalidationCallback*>::iterator iter = m_invalidationCallbacks.begin(); iter != end; ++iter)
            (**iter)(this);

        m_invalidationCallbacks.clear();
    }

    ProtectCountSet::iterator end = m_protectCountSet.end();
    for (ProtectCountSet::iterator it = m_protectCountSet.begin(); it != end; ++it)
        JSC::gcUnprotect(it->key);
    m_protectCountSet.clear();

    rootObjectSet().remove(this);
}

void RootObject::gcProtect(JSObject* jsObject)
{
    ASSERT(m_isValid);
    
    if (!m_protectCountSet.contains(jsObject)) {
        JSC::JSLockHolder holder(&globalObject()->vm());
        JSC::gcProtect(jsObject);
    }
    m_protectCountSet.add(jsObject);
}

void RootObject::gcUnprotect(JSObject* jsObject)
{
    ASSERT(m_isValid);
    
    if (!jsObject)
        return;

    if (m_protectCountSet.count(jsObject) == 1) {
        JSC::JSLockHolder holder(&globalObject()->vm());
        JSC::gcUnprotect(jsObject);
    }
    m_protectCountSet.remove(jsObject);
}

bool RootObject::gcIsProtected(JSObject* jsObject)
{
    ASSERT(m_isValid);
    return m_protectCountSet.contains(jsObject);
}

const void* RootObject::nativeHandle() const 
{ 
    ASSERT(m_isValid);
    return m_nativeHandle; 
}

JSGlobalObject* RootObject::globalObject() const
{
    ASSERT(m_isValid);
    return m_globalObject.get();
}

void RootObject::updateGlobalObject(JSGlobalObject* globalObject)
{
    m_globalObject.set(globalObject->vm(), globalObject);
}

void RootObject::addRuntimeObject(VM&, RuntimeObject* object)
{
    ASSERT(m_isValid);
    weakAdd(m_runtimeObjects, object, JSC::Weak<RuntimeObject>(object, this));
}

void RootObject::removeRuntimeObject(RuntimeObject* object)
{
    if (!m_isValid)
        return;
    weakRemove(m_runtimeObjects, object, object);
}

void RootObject::finalize(JSC::Handle<JSC::Unknown> handle, void*)
{
    RuntimeObject* object = static_cast<RuntimeObject*>(handle.slot()->asCell());

    Ref<RootObject> protectedThis(*this);
    object->invalidate();
    weakRemove(m_runtimeObjects, object, object);
}

} } // namespace JSC::Bindings

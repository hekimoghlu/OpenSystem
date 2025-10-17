/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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

#include "DOMWrapperWorld.h"
#include "JSDOMWrapper.h"
#include <JavaScriptCore/JSCJSValueInlines.h>
#include <JavaScriptCore/SlotVisitor.h>
#include <JavaScriptCore/WeakInlines.h>
#include <variant>

namespace WebCore {

// This class includes a lot of subtle GC related things, and changing this class can easily cause GC crashes.
// Any changes to this class must be reviewed by JavaScriptCore reviewers too.
class JSValueInWrappedObject {
    // This must be neither copyable nor movable. Changing this will break concurrent GC.
    WTF_MAKE_NONCOPYABLE(JSValueInWrappedObject);
    WTF_MAKE_NONMOVABLE(JSValueInWrappedObject);
public:
    JSValueInWrappedObject(JSC::JSValue = { });

    explicit operator bool() const;
    template<typename Visitor> void visit(Visitor&) const;
    void clear();

    // If you expect the value you store to be returned by getValue and not cleared under you, you *MUST* use set not setWeakly.
    // The owner parameter is typically the wrapper of the DOM node this class is embedded into but can be any GCed object that
    // will visit this JSValueInWrappedObject via visitAdditionalChildren/isReachableFromOpaqueRoots.
    void set(JSC::VM&, const JSC::JSCell* owner, JSC::JSValue);
    // Only use this if you actually expect this value to be weakly held. If you call visit on this value *DONT* set using setWeakly
    // use set instead. The GC might or might not keep your value around in that case.
    void setWeakly(JSC::JSValue);
    JSC::JSValue getValue(JSC::JSValue nullValue = JSC::jsUndefined()) const;

private:
    // Keep in mind that all of these fields are accessed concurrently without lock from concurrent GC thread.
    JSC::JSValue m_nonCell { };
    JSC::Weak<JSC::JSCell> m_cell { };
};

JSC::JSValue cachedPropertyValue(JSC::ThrowScope&, JSC::JSGlobalObject&, const JSDOMObject& owner, JSValueInWrappedObject& cacheSlot, const auto&);

inline JSValueInWrappedObject::JSValueInWrappedObject(JSC::JSValue value)
{
    setWeakly(value);
}

inline JSC::JSValue JSValueInWrappedObject::getValue(JSC::JSValue nullValue) const
{
    if (m_nonCell)
        return m_nonCell;
    return m_cell ? m_cell.get() : nullValue;
}

inline JSValueInWrappedObject::operator bool() const
{
    return m_nonCell || m_cell;
}

template<typename Visitor>
inline void JSValueInWrappedObject::visit(Visitor& visitor) const
{
    visitor.append(m_cell);
}

template void JSValueInWrappedObject::visit(JSC::AbstractSlotVisitor&) const;
template void JSValueInWrappedObject::visit(JSC::SlotVisitor&) const;

inline void JSValueInWrappedObject::setWeakly(JSC::JSValue value)
{
    if (!value.isCell()) {
        m_nonCell = value;
        m_cell.clear();
        return;
    }
    m_nonCell = { };
    JSC::Weak weak { value.asCell() };
    WTF::storeStoreFence();
    m_cell = WTFMove(weak);
}

inline void JSValueInWrappedObject::set(JSC::VM& vm, const JSC::JSCell* owner, JSC::JSValue value)
{
    setWeakly(value);
    vm.writeBarrier(owner, value);
}

inline void JSValueInWrappedObject::clear()
{
    m_nonCell = { };
    m_cell.clear();
}

inline JSC::JSValue cachedPropertyValue(JSC::ThrowScope& throwScope, JSC::JSGlobalObject& lexicalGlobalObject, const JSDOMObject& owner, JSValueInWrappedObject& cachedValue, const auto& function)
{
    if (cachedValue && isWorldCompatible(lexicalGlobalObject, cachedValue.getValue()))
        return cachedValue.getValue();

    auto value = function(throwScope);
    RETURN_IF_EXCEPTION(throwScope, { });

    cachedValue.set(lexicalGlobalObject.vm(), &owner, cloneAcrossWorlds(lexicalGlobalObject, owner, value));
    ASSERT(isWorldCompatible(lexicalGlobalObject, cachedValue.getValue()));
    return cachedValue.getValue();
}

} // namespace WebCore

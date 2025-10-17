/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 3, 2024.
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

#include "ActiveDOMCallback.h"
#include "JSDOMGlobalObject.h"
#include <JavaScriptCore/HeapInlines.h>
#include <JavaScriptCore/JSCell.h>
#include <JavaScriptCore/SlotVisitorInlines.h>
#include <JavaScriptCore/StrongInlines.h>

namespace WebCore {

class WEBCORE_EXPORT DOMGuardedObject : public RefCounted<DOMGuardedObject>, public ActiveDOMCallback {
public:
    ~DOMGuardedObject();

    bool isSuspended() const { return !m_guarded || !canInvokeCallback(); } // The wrapper world has gone away or active DOM objects have been suspended.

    template<typename Visitor> void visitAggregate(Visitor& visitor) { visitor.append(m_guarded); }

    JSC::JSValue guardedObject() const { return m_guarded.get(); }
    JSDOMGlobalObject* globalObject() const { return m_globalObject.get(); }

    void clear();

protected:
    DOMGuardedObject(JSDOMGlobalObject&, JSC::JSCell&);

    void contextDestroyed() override;
    bool isEmpty() const { return !m_guarded; }

    JSC::Weak<JSC::JSCell> m_guarded;
    JSC::Weak<JSDOMGlobalObject> m_globalObject;

private:
    void removeFromGlobalObject();
};

template <typename T> class DOMGuarded : public DOMGuardedObject {
protected:
    DOMGuarded(JSDOMGlobalObject& globalObject, T& guarded) : DOMGuardedObject(globalObject, guarded) { }
    T* guarded() const { return JSC::jsDynamicCast<T*>(guardedObject()); }
};

} // namespace WebCore

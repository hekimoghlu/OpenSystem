/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 3, 2022.
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
#include <wtf/Compiler.h>
#include <wtf/Forward.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class WindowProxy;

typedef UncheckedKeyHashMap<void*, JSC::Weak<JSC::JSObject>> DOMObjectWrapperMap;

class DOMWrapperWorld : public RefCounted<DOMWrapperWorld>, public CanMakeSingleThreadWeakPtr<DOMWrapperWorld> {
public:
    enum class Type {
        Normal,   // Main (e.g. Page)
        User,     // User Scripts (e.g. Extensions)
        Internal, // WebKit Internal (e.g. Media Controls)
    };

    static Ref<DOMWrapperWorld> create(JSC::VM& vm, Type type = Type::Internal, const String& name = { })
    {
        return adoptRef(*new DOMWrapperWorld(vm, type, name));
    }
    WEBCORE_EXPORT ~DOMWrapperWorld();

    // Free as much memory held onto by this world as possible.
    WEBCORE_EXPORT void clearWrappers();

    void didCreateWindowProxy(WindowProxy* controller) { m_jsWindowProxies.add(controller); }
    void didDestroyWindowProxy(WindowProxy* controller) { m_jsWindowProxies.remove(controller); }

    void setAllowAutofill() { m_allowAutofill = true; }
    bool allowAutofill() const { return m_allowAutofill; }

    void setAllowElementUserInfo() { m_allowElementUserInfo = true; }
    bool allowElementUserInfo() const { return m_allowElementUserInfo; }

    void setShadowRootIsAlwaysOpen() { m_shadowRootIsAlwaysOpen = true; }
    bool shadowRootIsAlwaysOpen() const { return m_shadowRootIsAlwaysOpen; }

    void disableLegacyOverrideBuiltInsBehavior() { m_shouldDisableLegacyOverrideBuiltInsBehavior = true; }
    bool shouldDisableLegacyOverrideBuiltInsBehavior() const { return m_shouldDisableLegacyOverrideBuiltInsBehavior; }

    DOMObjectWrapperMap& wrappers() { return m_wrappers; }

    Type type() const { return m_type; }
    bool isNormal() const { return m_type == Type::Normal; }
    bool isUser() const { return m_type == Type::User; }

    const String& name() const { return m_name; }

    JSC::VM& vm() const { return m_vm; }

protected:
    DOMWrapperWorld(JSC::VM&, Type, const String& name);

private:
    JSC::VM& m_vm;
    UncheckedKeyHashSet<WindowProxy*> m_jsWindowProxies;
    DOMObjectWrapperMap m_wrappers;

    String m_name;
    Type m_type { Type::Internal };

    bool m_allowAutofill { false };
    bool m_allowElementUserInfo { false };
    bool m_shadowRootIsAlwaysOpen { false };
    bool m_shouldDisableLegacyOverrideBuiltInsBehavior { false };
};

DOMWrapperWorld& normalWorld(JSC::VM&);
WEBCORE_EXPORT DOMWrapperWorld& mainThreadNormalWorld();
inline Ref<DOMWrapperWorld> protectedMainThreadNormalWorld() { return mainThreadNormalWorld(); }

inline DOMWrapperWorld& debuggerWorld() { return mainThreadNormalWorld(); }
inline DOMWrapperWorld& pluginWorld() { return mainThreadNormalWorld(); }

DOMWrapperWorld& currentWorld(JSC::JSGlobalObject&);
DOMWrapperWorld& worldForDOMObject(JSC::JSObject&);

// Helper function for code paths that must not share objects across isolated DOM worlds.
bool isWorldCompatible(JSC::JSGlobalObject&, JSC::JSValue);

inline DOMWrapperWorld& currentWorld(JSC::JSGlobalObject& lexicalGlobalObject)
{
    return JSC::jsCast<JSDOMGlobalObject*>(&lexicalGlobalObject)->world();
}

inline DOMWrapperWorld& worldForDOMObject(JSC::JSObject& object)
{
    return JSC::jsCast<JSDOMGlobalObject*>(object.globalObject())->world();
}

inline bool isWorldCompatible(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value)
{
    return !value.isObject() || &worldForDOMObject(*value.getObject()) == &currentWorld(lexicalGlobalObject);
}

} // namespace WebCore

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 25, 2023.
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

#include "Exception.h"
#include "InjectedScript.h"
#include "InspectorEnvironment.h"
#include <wtf/Expected.h>
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/NakedPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace JSC {
class CallFrame;
}

namespace Inspector {

class InjectedScriptHost;

class InjectedScriptManager {
    WTF_MAKE_NONCOPYABLE(InjectedScriptManager);
    WTF_MAKE_TZONE_ALLOCATED(InjectedScriptManager);
public:
    JS_EXPORT_PRIVATE InjectedScriptManager(InspectorEnvironment&, Ref<InjectedScriptHost>&&);
    JS_EXPORT_PRIVATE virtual ~InjectedScriptManager();

    JS_EXPORT_PRIVATE virtual void connect();
    JS_EXPORT_PRIVATE virtual void disconnect();
    JS_EXPORT_PRIVATE virtual void discardInjectedScripts();

    InjectedScriptHost& injectedScriptHost();
    InspectorEnvironment& inspectorEnvironment() const { return m_environment; }

    JS_EXPORT_PRIVATE InjectedScript injectedScriptFor(JSC::JSGlobalObject*);
    JS_EXPORT_PRIVATE InjectedScript injectedScriptForId(int);
    JS_EXPORT_PRIVATE int injectedScriptIdFor(JSC::JSGlobalObject*);
    JS_EXPORT_PRIVATE InjectedScript injectedScriptForObjectId(const String& objectId);
    void releaseObjectGroup(const String& objectGroup);
    void clearEventValue();
    void clearExceptionValue();

protected:
    virtual void didCreateInjectedScript(const InjectedScript&);

    UncheckedKeyHashMap<int, InjectedScript> m_idToInjectedScript;
    UncheckedKeyHashMap<JSC::JSGlobalObject*, int> m_scriptStateToId;

private:
    Expected<JSC::JSObject*, NakedPtr<JSC::Exception>> createInjectedScript(JSC::JSGlobalObject*, int id);

    InspectorEnvironment& m_environment;
    Ref<InjectedScriptHost> m_injectedScriptHost;
    int m_nextInjectedScriptId;
};

} // namespace Inspector

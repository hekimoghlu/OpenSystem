/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 11, 2022.
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

#include "JSShadowRealmGlobalScopeBase.h"
#include <JavaScriptCore/Weak.h>
#include <memory>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class JSDOMGlobalObject;
class ScriptExecutionContext;
class ScriptModuleLoader;

class ShadowRealmGlobalScope : public RefCounted<ShadowRealmGlobalScope> {
    friend class JSShadowRealmGlobalScopeBase;
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ShadowRealmGlobalScope);

public:
    static Ref<ShadowRealmGlobalScope> create(JSDOMGlobalObject*, ScriptModuleLoader*);
    ~ShadowRealmGlobalScope();

    ShadowRealmGlobalScope& self();
    ScriptModuleLoader& moduleLoader();
    JSShadowRealmGlobalScopeBase* wrapper();

protected:
    ShadowRealmGlobalScope(JSDOMGlobalObject*, ScriptModuleLoader*);

private:
    JSC::Weak<JSDOMGlobalObject> m_incubatingWrapper;
    ScriptModuleLoader* m_parentLoader { nullptr };
    JSC::Weak<JSShadowRealmGlobalScopeBase> m_wrapper;
    std::unique_ptr<ScriptModuleLoader> m_moduleLoader;
};

inline ShadowRealmGlobalScope& ShadowRealmGlobalScope::self()
{
    return *this;
}

inline JSShadowRealmGlobalScopeBase* ShadowRealmGlobalScope::wrapper()
{
    return m_wrapper.get();
}

} // namespace WebCore

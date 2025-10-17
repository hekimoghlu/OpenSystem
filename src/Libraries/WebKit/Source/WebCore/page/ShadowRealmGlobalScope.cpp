/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 24, 2023.
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
#include "ShadowRealmGlobalScope.h"

#include "JSDOMGlobalObject.h"
#include "JSShadowRealmGlobalScope.h"
#include "ScriptModuleLoader.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ShadowRealmGlobalScope);

Ref<ShadowRealmGlobalScope> ShadowRealmGlobalScope::create(JSDOMGlobalObject* wrapper, ScriptModuleLoader* loader)
{
    return adoptRef(*new ShadowRealmGlobalScope(wrapper, loader));
}

ShadowRealmGlobalScope::ShadowRealmGlobalScope(JSDOMGlobalObject* wrapper, ScriptModuleLoader* loader)
    : m_incubatingWrapper(wrapper)
    , m_parentLoader(loader)
{
}

ScriptModuleLoader& ShadowRealmGlobalScope::moduleLoader()
{
    if (m_moduleLoader)
        return *m_moduleLoader;

    auto wrapper = m_wrapper.get();
    ASSERT(wrapper);

    m_moduleLoader = m_parentLoader->shadowRealmLoader(wrapper).moveToUniquePtr();
    return *m_moduleLoader;
}

ShadowRealmGlobalScope::~ShadowRealmGlobalScope() = default;

} // namespace WebCore

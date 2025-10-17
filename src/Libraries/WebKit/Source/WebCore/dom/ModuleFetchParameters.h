/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 27, 2022.
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

#include <JavaScriptCore/ScriptFetchParameters.h>

namespace WebCore {

class ModuleFetchParameters : public JSC::ScriptFetchParameters {
public:
    static Ref<ModuleFetchParameters> create(JSC::ScriptFetchParameters::Type type, const String& integrity, bool isTopLevelModule)
    {
        return adoptRef(*new ModuleFetchParameters(type, integrity, isTopLevelModule));
    }

    const String& integrity() const override { return m_integrity; }
    bool isTopLevelModule() const override { return m_isTopLevelModule; }

private:
    ModuleFetchParameters(JSC::ScriptFetchParameters::Type type, const String& integrity, bool isTopLevelModule)
        : JSC::ScriptFetchParameters(type)
        , m_integrity(integrity)
        , m_isTopLevelModule(isTopLevelModule)
    {
    }

    String m_integrity;
    bool m_isTopLevelModule;
};

} // namespace WebCore

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 5, 2023.
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
#include "Module.h"

#include <shlwapi.h>
#include <wtf/text/CString.h>

namespace WebKit {

bool Module::load()
{
    ASSERT(!::PathIsRelativeW(m_path.wideCharacters().data()));
    m_module = ::LoadLibraryExW(m_path.wideCharacters().data(), 0, LOAD_WITH_ALTERED_SEARCH_PATH);
    return m_module;
}

void Module::unload()
{
    if (!m_module)
        return;
    ::FreeLibrary(m_module);
    m_module = 0;
}

void* Module::platformFunctionPointer(const char* functionName) const
{
    if (!m_module)
        return 0;
    auto proc = ::GetProcAddress(m_module, functionName);
    return reinterpret_cast<void*>(proc);
}

}

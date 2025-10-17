/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 28, 2024.
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

#include <gmodule.h>
#include <wtf/text/CString.h>

namespace WebKit {

bool Module::load()
{
    m_handle = g_module_open(m_path.utf8().data(), G_MODULE_BIND_LAZY);
    if (!m_handle)
        WTFLogAlways("Error loading module '%s': %s", m_path.utf8().data(), g_module_error());
    return m_handle;
}

void Module::unload()
{
    if (m_handle)
        g_module_close(m_handle);
}

void* Module::platformFunctionPointer(const char* functionName) const
{
    gpointer symbol = 0;
    g_module_symbol(m_handle, functionName, &symbol);
    return symbol;
}

}

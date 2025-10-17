/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 13, 2023.
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
#include <wtf/TranslatedProcess.h>

#if HAVE(CPU_TRANSLATION_CAPABILITY)

#include <mutex>
#include <sys/sysctl.h>

namespace WTF {

bool isX86BinaryRunningOnARM()
{
    static bool result;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [&] {
        int value = 0;
        size_t size = sizeof(value);
        if (sysctlbyname("sysctl.proc_translated", &value, &size, nullptr, 0) < 0)
            return;
        result = !!value;        
    });
    return result;
}

}

#endif

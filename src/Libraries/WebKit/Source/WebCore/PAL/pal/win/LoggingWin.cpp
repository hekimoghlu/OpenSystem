/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 8, 2023.
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
#include "Logging.h"

#if !LOG_DISABLED || !RELEASE_LOG_DISABLED

#include <windows.h>
#include <wtf/StdLibExtras.h>
#include <wtf/text/WTFString.h>

namespace PAL {

String logLevelString()
{
#if !LOG_DISABLED
    static constexpr const char* loggingEnvironmentVariable = "WebCoreLogging";
    DWORD length = GetEnvironmentVariableA(loggingEnvironmentVariable, 0, 0);
    if (!length)
        return emptyString();

    Vector<char> buffer(length);

    if (!GetEnvironmentVariableA(loggingEnvironmentVariable, buffer.data(), length))
        return emptyString();

    return String::fromLatin1(buffer.data());
#else
    return String();
#endif
}

}

#endif // !LOG_DISABLED || !RELEASE_LOG_DISABLED

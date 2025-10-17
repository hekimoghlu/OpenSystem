/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 17, 2024.
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

#include "LogInitialization.h"
#include <string.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>

namespace WTF {

String logLevelString()
{
    char* logEnv = getenv("WEBKIT_DEBUG");

    // Disable all log channels if WEBKIT_DEBUG is unset.
    if (!logEnv || !*logEnv)
        return makeString("-all"_s);

    // We set up the logs anyway because some of our logging, such as Soup's is available in release builds.
#if defined(NDEBUG) && RELEASE_LOG_DISABLED
    WTFLogAlways("WEBKIT_DEBUG is not empty, but this is a release build. Notice that many log messages will only appear in a debug build.");
#endif

    return String::fromLatin1(logEnv);
}

} // namespace WTF

#endif // !LOG_DISABLED || !RELEASE_LOG_DISABLED

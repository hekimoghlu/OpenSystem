/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 7, 2023.
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
#ifndef NotImplemented_h
#define NotImplemented_h

#include <wtf/Assertions.h>

#if PLATFORM(GTK)
    #define suppressNotImplementedWarning() getenv("DISABLE_NI_WARNING")
#else
    #define suppressNotImplementedWarning() false
#endif

#if LOG_DISABLED
    #define notImplemented() ((void)0)
#else

namespace WebCore {
WEBCORE_EXPORT WTFLogChannel* notImplementedLoggingChannel();
}

#define notImplemented() do { \
        static bool havePrinted = false; \
        if (!havePrinted && !suppressNotImplementedWarning()) { \
            WTFLogVerbose(__FILE__, __LINE__, WTF_PRETTY_FUNCTION, WebCore::notImplementedLoggingChannel(), "UNIMPLEMENTED: "); \
            havePrinted = true; \
        } \
    } while (0)

#endif // LOG_DISABLED

#endif // NotImplemented_h

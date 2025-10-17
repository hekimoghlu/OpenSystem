/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 20, 2022.
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
#include "CrashReporter.h"

#include <wtf/spi/cocoa/CrashReporterClientSPI.h>

#ifdef CRASHREPORTER_ANNOTATIONS_INITIALIZER
CRASHREPORTER_ANNOTATIONS_INITIALIZER()
#else
// Avoid having to link with libCrashReporterClient.a
CRASH_REPORTER_CLIENT_HIDDEN
struct crashreporter_annotations_t gCRAnnotations
    __attribute__((section("__DATA," CRASHREPORTER_ANNOTATIONS_SECTION)))
    = { CRASHREPORTER_ANNOTATIONS_VERSION, 0, 0, 0, 0, 0, 0, 0 };
#endif // CRASHREPORTER_ANNOTATIONS_INITIALIZER

namespace WTF {
void setCrashLogMessage(const char* message)
{
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    // We have to copy the string because CRSetCrashLogMessage doesn't.
    char* copiedMessage = message ? strdup(message) : nullptr;
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

    CRSetCrashLogMessage(copiedMessage);

    // Delete the message from last time, so we don't keep leaking messages.
    static char* previousCopiedCrashLogMessage;
    std::free(std::exchange(previousCopiedCrashLogMessage, copiedMessage));
}
}

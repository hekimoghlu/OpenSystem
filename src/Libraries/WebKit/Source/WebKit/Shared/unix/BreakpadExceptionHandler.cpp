/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 2, 2022.
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
#include "BreakpadExceptionHandler.h"

#if ENABLE(BREAKPAD)

#include <breakpad/client/linux/handler/exception_handler.h>
#include <mutex>
#include <signal.h>
#include <wtf/FileSystem.h>
#include <wtf/NeverDestroyed.h>

namespace WebKit {

void installBreakpadExceptionHandler()
{
    static std::once_flag onceFlag;
    static MainThreadLazyNeverDestroyed<google_breakpad::ExceptionHandler> exceptionHandler;
    static String breakpadMinidumpDir = String::fromUTF8(getenv("BREAKPAD_MINIDUMP_DIR"));

#ifdef BREAKPAD_MINIDUMP_DIR
    if (breakpadMinidumpDir.isEmpty())
        breakpadMinidumpDir = StringImpl::createFromCString(BREAKPAD_MINIDUMP_DIR);
#endif

    if (breakpadMinidumpDir.isEmpty())
        return;

    if (FileSystem::fileType(breakpadMinidumpDir) != FileSystem::FileType::Directory) {
        WTFLogAlways("Breakpad dir \"%s\" is not a directory, not installing handler", breakpadMinidumpDir.utf8().data());
        return;
    }

    std::call_once(onceFlag, []() {
        exceptionHandler.construct(google_breakpad::MinidumpDescriptor(breakpadMinidumpDir.utf8().data()), nullptr,
            [](const google_breakpad::MinidumpDescriptor&, void*, bool succeeded) -> bool {
                return succeeded;
            }, nullptr, true, -1);
    });
}
}
#endif


/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 20, 2025.
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

#include <wtf/Assertions.h>
#include <wtf/MainThread.h>
#include <wtf/ProcessID.h>

#define SLEEP_THREAD_FOR_DEBUGGER() \
do { \
    WTFReportError(__FILE__, __LINE__, WTF_PRETTY_FUNCTION, "Sleeping thread for debugger; attach to process (PID: %d) to unsleep the thread.", getCurrentProcessID()); \
    do { \
        sleep(1); \
        if (WTFIsDebuggerAttached()) \
            break; \
    } while (1); \
    WTFBreakpointTrap(); \
} while (0)

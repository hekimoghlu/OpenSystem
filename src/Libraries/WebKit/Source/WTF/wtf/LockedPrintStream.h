/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 23, 2022.
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

#include <wtf/PrintStream.h>
#include <wtf/RecursiveLockAdapter.h>
#include <wtf/WordLock.h>

namespace WTF {

// Makes every call to print() atomic.
class LockedPrintStream final : public PrintStream {
public:
    LockedPrintStream(std::unique_ptr<PrintStream> target);
    ~LockedPrintStream() final;
    
    void vprintf(const char* format, va_list) final WTF_ATTRIBUTE_PRINTF(2, 0);
    void flush() final;

private:
    PrintStream& begin() final;
    void end() final;

    // This needs to be a recursive lock because a printInternal or dump method could assert,
    // and that assert might want to log. Better to let it. This needs to be a WordLock so that
    // LockedPrintStream (i.e. cataLog) can be used to debug ParkingLot and Lock.
    RecursiveLockAdapter<WordLock> m_lock;
    std::unique_ptr<PrintStream> m_target;
};

} // namespace WTF

using WTF::LockedPrintStream;


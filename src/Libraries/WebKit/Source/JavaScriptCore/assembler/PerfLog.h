/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 29, 2022.
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

#if ENABLE(ASSEMBLER) && (OS(LINUX) || OS(DARWIN))

#include <stdio.h>
#include <wtf/Lock.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/CString.h>

namespace JSC {

class PerfLog {
    WTF_MAKE_TZONE_ALLOCATED(PerfLog);
    WTF_MAKE_NONCOPYABLE(PerfLog);
    friend class LazyNeverDestroyed<PerfLog>;
public:
    static void log(CString&&, const uint8_t* executableAddress, size_t);
    static void flush();

private:
    PerfLog();
    static PerfLog& singleton();

    void write(const AbstractLocker&, const void*, size_t) WTF_REQUIRES_LOCK(m_lock);
    void flush(const AbstractLocker&) WTF_REQUIRES_LOCK(m_lock);

    FILE* m_file { nullptr };
    void* m_marker { nullptr };
    uint64_t m_codeIndex { 0 };
    int m_fd { -1 };
    Lock m_lock;
};

} // namespace JSC

#endif  // ENABLE(ASSEMBLER) && (OS(LINUX) || OS(DARWIN))

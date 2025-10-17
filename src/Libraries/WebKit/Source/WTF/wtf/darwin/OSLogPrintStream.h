/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 11, 2025.
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

#if OS(DARWIN)

#include <wtf/Lock.h>
#include <wtf/PrintStream.h>
#include <wtf/RecursiveLockAdapter.h>
#include <wtf/text/CString.h>
#include <wtf/Vector.h>

#include <os/log.h>

namespace WTF {

class WTF_EXPORT_PRIVATE OSLogPrintStream final : public PrintStream {
public:
    OSLogPrintStream(os_log_t, os_log_type_t);
    ~OSLogPrintStream() final;
    
    static std::unique_ptr<OSLogPrintStream> open(const char* subsystem, const char* category, os_log_type_t = OS_LOG_TYPE_DEFAULT);
    
    void vprintf(const char* format, va_list) final WTF_ATTRIBUTE_PRINTF(2, 0);

private:
    os_log_t m_log;
    os_log_type_t m_logType;
    Lock m_stringLock;
    // We need a buffer because os_log doesn't wait for a new line to print the characters.
    CString m_string WTF_GUARDED_BY_LOCK(m_stringLock);
    size_t m_offset { 0 };
};

} // namespace WTF

using WTF::OSLogPrintStream;

#endif

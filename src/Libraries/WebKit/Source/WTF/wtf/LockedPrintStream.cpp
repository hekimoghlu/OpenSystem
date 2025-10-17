/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 23, 2024.
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
#include <wtf/LockedPrintStream.h>

namespace WTF {

LockedPrintStream::LockedPrintStream(std::unique_ptr<PrintStream> target)
    : m_target(WTFMove(target))
{
}

LockedPrintStream::~LockedPrintStream() = default;

void LockedPrintStream::vprintf(const char* format, va_list args)
{
    Locker locker { m_lock };
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    m_target->vprintf(format, args);
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
}

void LockedPrintStream::flush()
{
    Locker locker { m_lock };
    m_target->flush();
}

PrintStream& LockedPrintStream::begin()
{
    m_lock.lock();
    return *m_target;
}

void LockedPrintStream::end()
{
    m_lock.unlock();
}

} // namespace WTF


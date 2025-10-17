/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 12, 2024.
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

#include <utility>
#include <wtf/Compiler.h>
#include <wtf/Noncopyable.h>
#include <wtf/StdLibExtras.h>
#include <wtf/UniStdExtras.h>

namespace WTF {

class UnixFileDescriptor {
public:
    // This class is noncopyable because otherwise it's very hard to avoid accidental file
    // descriptor duplication. If you intentionally want a dup, call the duplicate method.
    WTF_MAKE_NONCOPYABLE(UnixFileDescriptor);

    UnixFileDescriptor() = default;

    enum AdoptionTag { Adopt };
    UnixFileDescriptor(int fd, AdoptionTag)
        : m_value(fd)
    { }

    enum DuplicationTag { Duplicate };
    UnixFileDescriptor(int fd, DuplicationTag)
    {
        if (fd >= 0)
            m_value = dupCloseOnExec(fd);
    }

    UnixFileDescriptor(UnixFileDescriptor&& o)
    {
        m_value = o.release();
    }

    UnixFileDescriptor& operator=(UnixFileDescriptor&& o)
    {
        if (&o == this)
            return *this;

        this->~UnixFileDescriptor();
        new (this) UnixFileDescriptor(WTFMove(o));
        return *this;
    }

    ~UnixFileDescriptor()
    {
        if (m_value >= 0)
            closeWithRetry(std::exchange(m_value, -1));
    }

    explicit operator bool() const { return m_value >= 0; }

    int value() const { return m_value; }

    UnixFileDescriptor duplicate() const
    {
        return UnixFileDescriptor { m_value, Duplicate };
    }

    int release() WARN_UNUSED_RETURN { return std::exchange(m_value, -1); }

private:
    int m_value { -1 };
};

} // namespace WTF

using WTF::UnixFileDescriptor;

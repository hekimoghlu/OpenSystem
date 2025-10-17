/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 22, 2023.
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

#if PLATFORM(GTK) || (PLATFORM(WPE) && ENABLE(WPE_PLATFORM))

#include <wtf/Function.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/unix/UnixFileDescriptor.h>

namespace WebKit {

class FenceMonitor {
    WTF_MAKE_NONCOPYABLE(FenceMonitor);
public:
    explicit FenceMonitor(Function<void()>&&);
    ~FenceMonitor();

    void addFileDescriptor(WTF::UnixFileDescriptor&&);
    bool hasFileDescriptor() const { return !!m_fd; }

private:
    void ensureSource();

    Function<void()> m_callback;
    WTF::UnixFileDescriptor m_fd;
    GRefPtr<GSource> m_source;
};

} // namespace WebKit

#endif // PLATFORM(GTK) || (PLATFORM(WPE) && ENABLE(WPE_PLATFORM))

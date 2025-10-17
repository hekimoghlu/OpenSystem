/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 14, 2023.
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
#include "FenceMonitor.h"

#if PLATFORM(GTK) || (PLATFORM(WPE) && ENABLE(WPE_PLATFORM))
#include <glib-unix.h>

#if PLATFORM(GTK)
#include <gtk/gtk.h>
#endif

namespace WebKit {

FenceMonitor::FenceMonitor(Function<void()>&& callback)
    : m_callback(WTFMove(callback))
{
}

FenceMonitor::~FenceMonitor()
{
    if (m_source)
        g_source_destroy(m_source.get());
}

struct FenceSource {
    static GSourceFuncs sourceFuncs;

    GSource base;
    gpointer tag;
};

GSourceFuncs FenceSource::sourceFuncs = {
    nullptr, // prepare
    nullptr, // check
    // dispatch
    [](GSource* base, GSourceFunc callback, gpointer userData) -> gboolean
    {
        auto& source = *reinterpret_cast<FenceSource*>(base);
        g_source_remove_unix_fd(&source.base, source.tag);
        source.tag = nullptr;

        callback(userData);
        return G_SOURCE_CONTINUE;
    },
    nullptr, // finalize
    nullptr, // closure_callback
    nullptr, // closure_marshall
};

void FenceMonitor::ensureSource()
{
    if (LIKELY(m_source))
        return;

    m_source = adoptGRef(g_source_new(&FenceSource::sourceFuncs, sizeof(FenceSource)));
    g_source_set_name(m_source.get(), "[WebKit] Fence monitor");
#if PLATFORM(GTK)
    g_source_set_priority(m_source.get(), GDK_PRIORITY_REDRAW - 1);
#endif
    g_source_set_callback(m_source.get(), [](gpointer userData) -> gboolean {
        auto& monitor = *static_cast<FenceMonitor*>(userData);
        monitor.m_fd = { };
        monitor.m_callback();
        return G_SOURCE_CONTINUE;
    }, this, nullptr);
    g_source_attach(m_source.get(), g_main_context_get_thread_default());
}

static bool isFileDescriptorReadable(int fd)
{
    GPollFD pollFD = { fd, G_IO_IN, 0 };
    if (!g_poll(&pollFD, 1, 0))
        return FALSE;

    return pollFD.revents & (G_IO_IN | G_IO_NVAL);
}

void FenceMonitor::addFileDescriptor(UnixFileDescriptor&& fd)
{
    RELEASE_ASSERT(!m_fd);

    if (!fd || isFileDescriptorReadable(fd.value())) {
        m_callback();
        return;
    }

    m_fd = WTFMove(fd);

    ensureSource();
    auto& source = *reinterpret_cast<FenceSource*>(m_source.get());
    if (source.tag)
        g_source_remove_unix_fd(&source.base, source.tag);
    source.tag = g_source_add_unix_fd(&source.base, m_fd.value(), G_IO_IN);
}

} // namespace WebKit

#endif // PLATFORM(GTK) || (PLATFORM(WPE) && ENABLE(WPE_PLATFORM))

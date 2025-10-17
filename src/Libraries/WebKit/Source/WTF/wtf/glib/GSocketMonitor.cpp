/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 9, 2022.
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
#include "GSocketMonitor.h"

#include <gio/gio.h>
#include <wtf/glib/RunLoopSourcePriority.h>

namespace WTF {

GSocketMonitor::~GSocketMonitor()
{
    RELEASE_ASSERT(!m_isExecutingCallback);
    stop();
}

gboolean GSocketMonitor::socketSourceCallback(GSocket*, GIOCondition condition, GSocketMonitor* monitor)
{
    if (g_cancellable_is_cancelled(monitor->m_cancellable.get()))
        return G_SOURCE_REMOVE;

    monitor->m_isExecutingCallback = true;
    gboolean result = monitor->m_callback(condition);
    monitor->m_isExecutingCallback = false;

    if (monitor->m_shouldDestroyCallback) {
        // Destroying m_callback could also destroy this GSocketMonitor, so that has to be last.
        monitor->m_shouldDestroyCallback = false;
        monitor->m_callback = nullptr;
    }

    return result;
}

void GSocketMonitor::start(GSocket* socket, GIOCondition condition, RunLoop& runLoop, Function<gboolean(GIOCondition)>&& callback)
{
    stop();

    m_cancellable = adoptGRef(g_cancellable_new());
    m_source = adoptGRef(g_socket_create_source(socket, condition, m_cancellable.get()));
    g_source_set_name(m_source.get(), "[WebKit] Socket monitor");
    m_callback = WTFMove(callback);
    g_source_set_callback(m_source.get(), reinterpret_cast<GSourceFunc>(reinterpret_cast<GCallback>(socketSourceCallback)), this, nullptr);
    g_source_set_priority(m_source.get(), RunLoopSourcePriority::RunLoopDispatcher);
    g_source_attach(m_source.get(), runLoop.mainContext());
}

void GSocketMonitor::stop()
{
    if (!m_source)
        return;

    g_cancellable_cancel(m_cancellable.get());
    m_cancellable = nullptr;
    g_source_destroy(m_source.get());
    m_source = nullptr;

    // It's normal to stop the socket monitor from inside its callback.
    // Don't destroy the callback while it's still executing.
    if (m_isExecutingCallback)
        m_shouldDestroyCallback = true;
    else
        m_callback = nullptr;
}

} // namespace WTF

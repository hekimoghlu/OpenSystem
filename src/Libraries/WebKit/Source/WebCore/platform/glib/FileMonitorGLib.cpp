/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 25, 2024.
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
#include "FileMonitor.h"

#include <wtf/FileSystem.h>
#include <wtf/glib/GUniquePtr.h>

namespace WebCore {

FileMonitor::FileMonitor(const String& path, Ref<WorkQueue>&& handlerQueue, Function<void(FileChangeType)>&& modificationHandler)
    : m_handlerQueue(WTFMove(handlerQueue))
    , m_modificationHandler(WTFMove(modificationHandler))
{
    if (path.isEmpty() || !m_modificationHandler)
        return;

    Function<void ()> createPlatformMonitor = [&] {
        auto file = adoptGRef(g_file_new_for_path(FileSystem::fileSystemRepresentation(path).data()));
        GUniqueOutPtr<GError> error;
        m_platformMonitor = adoptGRef(g_file_monitor(file.get(), G_FILE_MONITOR_NONE, nullptr, &error.outPtr()));
        if (m_platformMonitor)
            g_signal_connect(m_platformMonitor.get(), "changed", G_CALLBACK(fileChangedCallback), this);
        else
            WTFLogAlways("Failed to create a monitor for path %s: %s", path.utf8().data(), error->message);
    };

    // The monitor can be created in the work queue thread.
    if (m_handlerQueue->isCurrent()) {
        createPlatformMonitor();
        return;
    }

    m_handlerQueue->dispatchSync([createPlatformMonitor = WTFMove(createPlatformMonitor)] {
        createPlatformMonitor();
    });
}

FileMonitor::~FileMonitor()
{
    // The monitor can be destroyed in the work queue thread.
    if (m_handlerQueue->isCurrent()) {
        cancel();
        return;
    }

    m_handlerQueue->dispatchSync([this] {
        cancel();
    });
}

void FileMonitor::fileChangedCallback(GFileMonitor*, GFile*, GFile*, GFileMonitorEvent event, FileMonitor* monitor)
{
    switch (event) {
    case G_FILE_MONITOR_EVENT_DELETED:
        monitor->didChange(FileChangeType::Removal);
        break;
    case G_FILE_MONITOR_EVENT_CHANGES_DONE_HINT:
    case G_FILE_MONITOR_EVENT_CREATED:
        monitor->didChange(FileChangeType::Modification);
        break;
    default:
        break;
    }
}

void FileMonitor::didChange(FileChangeType type)
{
    ASSERT(!isMainThread());
    if (type == FileChangeType::Removal)
        cancel();
    m_modificationHandler(type);
}

void FileMonitor::cancel()
{
    if (!m_platformMonitor)
        return;

    g_file_monitor_cancel(m_platformMonitor.get());
    m_platformMonitor = nullptr;
}

} // namespace WebCore

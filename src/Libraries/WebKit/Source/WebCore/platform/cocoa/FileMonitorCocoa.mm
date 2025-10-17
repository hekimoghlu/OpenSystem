/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 17, 2022.
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
#import "config.h"
#import "FileMonitor.h"

#import "Logging.h"
#import <wtf/BlockPtr.h>
#import <wtf/FileSystem.h>

namespace WebCore {
    
constexpr unsigned monitorMask = DISPATCH_VNODE_DELETE | DISPATCH_VNODE_WRITE | DISPATCH_VNODE_RENAME | DISPATCH_VNODE_REVOKE;
constexpr unsigned fileUnavailableMask = DISPATCH_VNODE_DELETE | DISPATCH_VNODE_RENAME | DISPATCH_VNODE_REVOKE;

FileMonitor::FileMonitor(const String& path, Ref<WorkQueue>&& handlerQueue, WTF::Function<void(FileChangeType)>&& modificationHandler)
{
    if (path.isEmpty())
        return;

    if (!modificationHandler)
        return;

    auto handle = FileSystem::openFile(path, FileSystem::FileOpenMode::EventsOnly);
    if (handle == FileSystem::invalidPlatformFileHandle) {
        RELEASE_LOG_ERROR(ResourceLoadStatistics, "Failed to open statistics file for monitoring: %s", path.utf8().data());
        return;
    }

    // The source (platformMonitor) retains the dispatch queue.
    m_platformMonitor = adoptOSObject(dispatch_source_create(DISPATCH_SOURCE_TYPE_VNODE, handle, monitorMask, handlerQueue->dispatchQueue()));

    LOG(ResourceLoadStatistics, "Creating monitor %p", m_platformMonitor.get());

    dispatch_source_set_event_handler(m_platformMonitor.get(), makeBlockPtr([modificationHandler = WTFMove(modificationHandler), fileMonitor = m_platformMonitor] {
        // If this is getting called after the monitor was cancelled, just drop the notification.
        if (dispatch_source_testcancel(fileMonitor.get()))
            return;

        unsigned flag = dispatch_source_get_data(fileMonitor.get());
        LOG(ResourceLoadStatistics, "File event %#X for monitor %p", flag, fileMonitor.get());
        if (flag & fileUnavailableMask) {
            modificationHandler(FileChangeType::Removal);
            dispatch_source_cancel(fileMonitor.get());
        } else {
            ASSERT(flag & DISPATCH_VNODE_WRITE);
            modificationHandler(FileChangeType::Modification);
        }
    }).get());
    
    dispatch_source_set_cancel_handler(m_platformMonitor.get(), [handle] () mutable {
        FileSystem::closeFile(handle);
    });
    
    dispatch_resume(m_platformMonitor.get());
}

FileMonitor::~FileMonitor()
{
    if (!m_platformMonitor)
        return;

    LOG(ResourceLoadStatistics, "Stopping monitor %p", m_platformMonitor.get());

    dispatch_source_cancel(m_platformMonitor.get());
}

} // namespace WebCore

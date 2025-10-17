/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 29, 2022.
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

#include <wtf/Function.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WorkQueue.h>
#include <wtf/text/WTFString.h>

#if USE(COCOA_EVENT_LOOP)
#include <dispatch/dispatch.h>
#include <wtf/OSObjectPtr.h>
#endif

#if USE(GLIB)
#include <gio/gio.h>
#include <wtf/glib/GRefPtr.h>
#endif

namespace WebCore {

class FileMonitor {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(FileMonitor, WEBCORE_EXPORT);
public:
    enum class FileChangeType { Modification, Removal };

    WEBCORE_EXPORT FileMonitor(const String&, Ref<WorkQueue>&& handlerQueue, Function<void(FileChangeType)>&& modificationHandler);
    WEBCORE_EXPORT ~FileMonitor();

private:
#if USE(COCOA_EVENT_LOOP)
    OSObjectPtr<dispatch_source_t> m_platformMonitor;
#endif
#if USE(GLIB)
    static void fileChangedCallback(GFileMonitor*, GFile*, GFile*, GFileMonitorEvent, FileMonitor*);
    void didChange(FileChangeType);
    void cancel();
    Ref<WorkQueue> m_handlerQueue;
    Function<void(FileChangeType)> m_modificationHandler;
    GRefPtr<GFileMonitor> m_platformMonitor;
#endif
};

} // namespace WebCore

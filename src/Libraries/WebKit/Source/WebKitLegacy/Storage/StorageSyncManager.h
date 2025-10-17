/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 4, 2024.
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
#ifndef StorageSyncManager_h
#define StorageSyncManager_h

#include <functional>
#include <wtf/Forward.h>
#include <wtf/Function.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class StorageThread;
class StorageAreaSync;

class StorageSyncManager : public RefCounted<StorageSyncManager> {
public:
    static Ref<StorageSyncManager> create(const String& path);
    ~StorageSyncManager();

    void dispatch(Function<void ()>&&);
    void close();

private:
    explicit StorageSyncManager(const String& path);

    std::unique_ptr<StorageThread> m_thread;

// The following members are subject to thread synchronization issues
public:
    // To be called from the background thread:
    String fullDatabaseFilename(const String& databaseIdentifier);

private:
    String m_path;
};

} // namespace WebCore

#endif // StorageSyncManager_h

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 20, 2023.
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
#include "FileSystemStorageHandleRegistry.h"

#include "FileSystemStorageHandle.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FileSystemStorageHandleRegistry);

Ref<FileSystemStorageHandleRegistry> FileSystemStorageHandleRegistry::create()
{
    return adoptRef(*new FileSystemStorageHandleRegistry);
}

FileSystemStorageHandleRegistry::FileSystemStorageHandleRegistry() = default;

void FileSystemStorageHandleRegistry::registerHandle(WebCore::FileSystemHandleIdentifier identifier, FileSystemStorageHandle& handle)
{
    ASSERT(!m_handles.contains(identifier));

    m_handles.add(identifier, handle);
}

void FileSystemStorageHandleRegistry::unregisterHandle(WebCore::FileSystemHandleIdentifier identifier)
{
    ASSERT(m_handles.contains(identifier));

    m_handles.remove(identifier);
}

FileSystemStorageHandle* FileSystemStorageHandleRegistry::getHandle(WebCore::FileSystemHandleIdentifier identifier)
{
    return m_handles.get(identifier).get();
}

} // namespace WebKit


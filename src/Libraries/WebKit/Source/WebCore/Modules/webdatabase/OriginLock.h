/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 29, 2023.
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

#include <wtf/FileSystem.h>
#include <wtf/Lock.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class OriginLock : public ThreadSafeRefCounted<OriginLock> {
public:
    static Ref<OriginLock> create(const String& originPath)
    {
        return adoptRef(*new OriginLock(originPath));
    }
    WEBCORE_EXPORT ~OriginLock();

    void lock();
    void unlock();

    static void deleteLockFile(const String& originPath);

private:
    explicit OriginLock(const String& originPath);

    String m_lockFileName;
    Lock m_mutex;
#if USE(FILE_LOCK)
    FileSystem::PlatformFileHandle m_lockHandle { FileSystem::invalidPlatformFileHandle };
#endif
};

} // namespace WebCore

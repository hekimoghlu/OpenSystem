/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 17, 2022.
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

#include <wtf/Markable.h>
#include <wtf/Threading.h>
#include <wtf/WallTime.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class DatabaseDetails {
public:
    DatabaseDetails() = default;

    DatabaseDetails(const DatabaseDetails& details)
        : m_name(details.m_name)
        , m_displayName(details.m_displayName)
        , m_expectedUsage(details.m_expectedUsage)
        , m_currentUsage(details.m_currentUsage)
        , m_creationTime(details.m_creationTime)
        , m_modificationTime(details.m_modificationTime)
#if ASSERT_ENABLED
        , m_thread(details.m_thread.copyRef())
#endif
    {
    }

    DatabaseDetails& operator=(const DatabaseDetails& details)
    {
        m_name = details.m_name;
        m_displayName = details.m_displayName;
        m_expectedUsage = details.m_expectedUsage;
        m_currentUsage = details.m_currentUsage;
        m_creationTime = details.m_creationTime;
        m_modificationTime = details.m_modificationTime;
#if ASSERT_ENABLED
        m_thread = details.m_thread.copyRef();
#endif
        return *this;
    }

    DatabaseDetails(const String& databaseName, const String& displayName, uint64_t expectedUsage, uint64_t currentUsage, std::optional<WallTime> creationTime, std::optional<WallTime> modificationTime)
        : m_name(databaseName)
        , m_displayName(displayName)
        , m_expectedUsage(expectedUsage)
        , m_currentUsage(currentUsage)
        , m_creationTime(creationTime)
        , m_modificationTime(modificationTime)
    {
    }

    const String& name() const { return m_name; }
    const String& displayName() const { return m_displayName; }
    uint64_t expectedUsage() const { return m_expectedUsage; }
    uint64_t currentUsage() const { return m_currentUsage; }
    std::optional<WallTime> creationTime() const { return m_creationTime; }
    std::optional<WallTime> modificationTime() const { return m_modificationTime; }
#if ASSERT_ENABLED
    Thread& thread() const { return m_thread.get(); }
#endif

private:
    String m_name;
    String m_displayName;
    uint64_t m_expectedUsage { 0 };
    uint64_t m_currentUsage { 0 };
    Markable<WallTime> m_creationTime;
    Markable<WallTime> m_modificationTime;
#if ASSERT_ENABLED
    Ref<Thread> m_thread { Thread::current() };
#endif
};

} // namespace WebCore

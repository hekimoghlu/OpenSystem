/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 23, 2023.
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
#include <wtf/OSAllocator.h>

#include <errno.h>
#include <memory-extra/vss.h>
#include <wtf/Assertions.h>
#include <wtf/DataLog.h>
#include <wtf/MathExtras.h>
#include <wtf/PageBlock.h>
#include <wtf/SafeStrerror.h>
#include <wtf/text/CString.h>

namespace WTF {

void* OSAllocator::tryReserveAndCommit(size_t bytes, Usage usage, bool writable, bool executable, bool jitCageEnabled, bool includesGuardPages)
{
    ASSERT_UNUSED(includesGuardPages, !includesGuardPages);
    ASSERT_UNUSED(jitCageEnabled, !jitCageEnabled);
    ASSERT_UNUSED(executable, !executable);

    void* result = memory_extra::vss::reserve(bytes);
    if (!result)
        return nullptr;

    bool success = memory_extra::vss::commit(result, bytes, writable, usage);
    if (!success) {
        memory_extra::vss::release(result, bytes);
        return nullptr;
    }

    return result;
}

void* OSAllocator::tryReserveUncommitted(size_t bytes, Usage usage, bool writable, bool executable, bool jitCageEnabled, bool includesGuardPages)
{
    UNUSED_PARAM(usage);
    UNUSED_PARAM(writable);
    ASSERT_UNUSED(includesGuardPages, !includesGuardPages);
    ASSERT_UNUSED(jitCageEnabled, !jitCageEnabled);
    ASSERT_UNUSED(executable, !executable);

    void* result = memory_extra::vss::reserve(bytes);
    if (!result)
        return nullptr;

    return result;
}

void* OSAllocator::reserveUncommitted(size_t bytes, Usage usage, bool writable, bool executable, bool jitCageEnabled, bool includesGuardPages)
{
    void* result = tryReserveUncommitted(bytes, usage, writable, executable, jitCageEnabled, includesGuardPages);
    RELEASE_ASSERT(result);
    return result;
}

void* OSAllocator::tryReserveUncommittedAligned(size_t bytes, size_t alignment, Usage usage, bool writable, bool executable, bool jitCageEnabled, bool includesGuardPages)
{
    ASSERT_UNUSED(includesGuardPages, !includesGuardPages);
    ASSERT_UNUSED(jitCageEnabled, !jitCageEnabled);
    ASSERT_UNUSED(executable, !executable);
    UNUSED_PARAM(usage);
    UNUSED_PARAM(writable);
    ASSERT(hasOneBitSet(alignment) && alignment >= pageSize());

    void* result = memory_extra::vss::reserve(bytes, alignment);
    if (!result)
        return nullptr;

    return result;
}

void* OSAllocator::reserveAndCommit(size_t bytes, Usage usage, bool writable, bool executable, bool jitCageEnabled, bool includesGuardPages)
{
    void* result = tryReserveAndCommit(bytes, usage, writable, executable, jitCageEnabled, includesGuardPages);
    RELEASE_ASSERT(result);
    return result;
}

void OSAllocator::commit(void* address, size_t bytes, bool writable, bool executable)
{
    ASSERT_UNUSED(executable, !executable);
    bool success = memory_extra::vss::commit(address, bytes, writable, -1);
    RELEASE_ASSERT(success);
}

void OSAllocator::decommit(void* address, size_t bytes)
{
    bool success = memory_extra::vss::decommit(address, bytes);
    RELEASE_ASSERT(success);
}

void OSAllocator::hintMemoryNotNeededSoon(void* address, size_t bytes)
{
    UNUSED_PARAM(address);
    UNUSED_PARAM(bytes);
}

void OSAllocator::releaseDecommitted(void* address, size_t bytes)
{
    bool success = memory_extra::vss::release(address, bytes);
    RELEASE_ASSERT(success);
}

bool OSAllocator::tryProtect(void* address, size_t bytes, bool readable, bool writable)
{
    return memory_extra::vss::protect(address, bytes, readable, writable, -1);
}

void OSAllocator::protect(void* address, size_t bytes, bool readable, bool writable)
{
    if (bool result = tryProtect(address, bytes, readable, writable); UNLIKELY(!result)) {
        dataLogLn("mprotect failed: ", safeStrerror(errno).data());
        RELEASE_ASSERT_NOT_REACHED();
    }
}

} // namespace WTF

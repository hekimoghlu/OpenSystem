/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 18, 2023.
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
#include <wtf/ObjectIdentifier.h>

#include <atomic>
#include <wtf/MainThread.h>
#include <wtf/PrintStream.h>

namespace WTF {

uint64_t ObjectIdentifierMainThreadAccessTraits<uint64_t>::generateIdentifierInternal()
{
    ASSERT(isMainThread()); // You should use AtomicObjectIdentifier if you're hitting this assertion.
    static uint64_t current = 0;
    return ++current;
}

TextStream& operator<<(TextStream& ts, const ObjectIdentifierGenericBase<uint64_t>& identifier)
{
    ts << identifier.toRawValue();
    return ts;
}

void printInternal(PrintStream& out, const ObjectIdentifierGenericBase<uint64_t>& identifier)
{
    out.print(identifier.toRawValue());
}

uint64_t ObjectIdentifierThreadSafeAccessTraits<uint64_t>::generateIdentifierInternal()
{
    static std::atomic<uint64_t> current;
    return ++current;
}

UUID ObjectIdentifierMainThreadAccessTraits<UUID>::generateIdentifierInternal()
{
    ASSERT(isMainThread()); // You should use AtomicObjectIdentifier if you're hitting this assertion.
    return UUID::createVersion4();
}

UUID ObjectIdentifierThreadSafeAccessTraits<UUID>::generateIdentifierInternal()
{
    return UUID::createVersion4();
}

TextStream& operator<<(TextStream& ts, const ObjectIdentifierGenericBase<UUID>& identifier)
{
    ts << identifier.toRawValue();
    return ts;
}

void printInternal(PrintStream& out, const ObjectIdentifierGenericBase<UUID>& identifier)
{
    out.print(identifier.toRawValue());
}

} // namespace WTF

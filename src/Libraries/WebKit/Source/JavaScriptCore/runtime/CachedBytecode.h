/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 16, 2022.
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

#include "CacheUpdate.h"
#include "LeafExecutable.h"
#include "ParserModes.h"
#include <wtf/MallocSpan.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>

namespace JSC {

class UnlinkedFunctionExecutable;

class CachedBytecode : public RefCounted<CachedBytecode> {
    WTF_MAKE_NONCOPYABLE(CachedBytecode);

public:
    static Ref<CachedBytecode> create()
    {
        return adoptRef(*new CachedBytecode(CachePayload::makeEmptyPayload()));
    }

    static Ref<CachedBytecode> create(FileSystem::MappedFileData&& data, LeafExecutableMap&& leafExecutables = { })
    {
        return adoptRef(*new CachedBytecode(CachePayload::makeMappedPayload(WTFMove(data)), WTFMove(leafExecutables)));
    }

    static Ref<CachedBytecode> create(MallocSpan<uint8_t, VMMalloc>&& data, LeafExecutableMap&& leafExecutables)
    {
        return adoptRef(*new CachedBytecode(CachePayload::makeMallocPayload(WTFMove(data)), WTFMove(leafExecutables)));
    }

    LeafExecutableMap& leafExecutables() { return m_leafExecutables; }

    JS_EXPORT_PRIVATE void addGlobalUpdate(Ref<CachedBytecode>);
    JS_EXPORT_PRIVATE void addFunctionUpdate(const UnlinkedFunctionExecutable*, CodeSpecializationKind, Ref<CachedBytecode>);

    using ForEachUpdateCallback = Function<void(off_t, std::span<const uint8_t>)>;
    JS_EXPORT_PRIVATE void commitUpdates(const ForEachUpdateCallback&) const;

    std::span<const uint8_t> span() const { return m_payload.span(); }
    size_t size() const { return m_payload.size(); }
    bool hasUpdates() const { return !m_updates.isEmpty(); }
    size_t sizeForUpdate() const { return m_size; }

private:
    CachedBytecode(CachePayload&& payload, LeafExecutableMap&& leafExecutables = { })
        : m_size(payload.size())
        , m_payload(WTFMove(payload))
        , m_leafExecutables(WTFMove(leafExecutables))
    {
    }

    void copyLeafExecutables(const CachedBytecode&);

    size_t m_size { 0 };
    CachePayload m_payload;
    LeafExecutableMap m_leafExecutables;
    Vector<CacheUpdate> m_updates;
};


} // namespace JSC

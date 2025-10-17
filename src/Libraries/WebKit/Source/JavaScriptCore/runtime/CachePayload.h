/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
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

#include "VM.h"
#include <variant>
#include <wtf/FileSystem.h>
#include <wtf/MallocSpan.h>

namespace JSC {

class CachePayload {
public:
    JS_EXPORT_PRIVATE static CachePayload makeMappedPayload(FileSystem::MappedFileData&&);
    JS_EXPORT_PRIVATE static CachePayload makeMallocPayload(MallocSpan<uint8_t, VMMalloc>&&);
    JS_EXPORT_PRIVATE static CachePayload makeEmptyPayload();

    JS_EXPORT_PRIVATE CachePayload(CachePayload&&);
    JS_EXPORT_PRIVATE ~CachePayload();

    size_t size() const { return span().size(); }
    JS_EXPORT_PRIVATE std::span<const uint8_t> span() const;

private:
    using DataType = std::variant<MallocSpan<uint8_t, VMMalloc>, FileSystem::MappedFileData>;
    explicit CachePayload(DataType&&);

    DataType m_data;
};

} // namespace JSC

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 7, 2024.
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

#include "ThreadSafeDataBuffer.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class SerializedScriptValue;

class IDBValue {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(IDBValue, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT IDBValue();
    IDBValue(const SerializedScriptValue&);
    WEBCORE_EXPORT IDBValue(const ThreadSafeDataBuffer&);
    IDBValue(const SerializedScriptValue&, const Vector<String>& blobURLs, const Vector<String>& blobFilePaths);
    WEBCORE_EXPORT IDBValue(const ThreadSafeDataBuffer&, Vector<String>&& blobURLs, Vector<String>&& blobFilePaths);
    IDBValue(const ThreadSafeDataBuffer&, const Vector<String>& blobURLs, const Vector<String>& blobFilePaths);

    void setAsIsolatedCopy(const IDBValue&);
    WEBCORE_EXPORT IDBValue isolatedCopy() const;

    const ThreadSafeDataBuffer& data() const { return m_data; }
    const Vector<String>& blobURLs() const { return m_blobURLs; }
    const Vector<String>& blobFilePaths() const { return m_blobFilePaths; }

    size_t size() const;
private:
    ThreadSafeDataBuffer m_data;
    Vector<String> m_blobURLs;
    Vector<String> m_blobFilePaths;
};

} // namespace WebCore

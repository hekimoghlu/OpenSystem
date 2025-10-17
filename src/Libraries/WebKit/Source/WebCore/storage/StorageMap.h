/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 19, 2025.
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

#include <wtf/HashMap.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

// This class uses copy-on-write semantics.
class StorageMap {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(StorageMap, WEBCORE_EXPORT);
public:
    // Quota size measured in bytes.
    WEBCORE_EXPORT explicit StorageMap(unsigned quotaSize);

    WEBCORE_EXPORT unsigned length() const;
    WEBCORE_EXPORT String key(unsigned index);
    WEBCORE_EXPORT String getItem(const String&) const;
    WEBCORE_EXPORT void setItem(const String& key, const String& value, String& oldValue, bool& quotaException);
    WEBCORE_EXPORT void setItemIgnoringQuota(const String& key, const String& value);
    WEBCORE_EXPORT void removeItem(const String&, String& oldValue);
    WEBCORE_EXPORT void clear();

    WEBCORE_EXPORT bool contains(const String& key) const;

    WEBCORE_EXPORT void importItems(HashMap<String, String>&&);
    const HashMap<String, String>& items() const { return m_impl->map; }

    unsigned quota() const { return m_quotaSize; }

    bool isShared() const { return !m_impl->hasOneRef(); }

    static constexpr unsigned noQuota = std::numeric_limits<unsigned>::max();

private:
    void invalidateIterator();
    void setIteratorToIndex(unsigned);

    struct Impl : public RefCounted<Impl> {
        static Ref<Impl> create()
        {
            return adoptRef(*new Impl);
        }

        Ref<Impl> copy() const;

        HashMap<String, String> map;
        HashMap<String, String>::iterator iterator { map.end() };
        unsigned iteratorIndex { std::numeric_limits<unsigned>::max() };
        unsigned currentSize { 0 };
    };

    Ref<Impl> m_impl;
    unsigned m_quotaSize { noQuota }; // Measured in bytes.
};

} // namespace WebCore

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 21, 2021.
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

#include <wtf/RefCounted.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

// FIXME: Some consumers of this class may benefit from lazily fetching items rather
//        than creating the list statically as is currently the only option.
class DOMStringList : public RefCounted<DOMStringList> {
public:
    static Ref<DOMStringList> create()
    {
        return adoptRef(*new DOMStringList);
    }

    static Ref<DOMStringList> create(Vector<String>&& strings)
    {
        return adoptRef(*new DOMStringList(WTFMove(strings)));
    }

    bool isEmpty() const { return m_strings.isEmpty(); }
    void clear() { m_strings.clear(); }
    void append(String&& string) { m_strings.append(WTFMove(string)); }
    void sort();

    bool isSupportedPropertyIndex(unsigned index) const { return index < m_strings.size(); }

    // Implements the IDL.
    size_t length() const { return m_strings.size(); }
    String item(unsigned index) const;
    bool contains(const String& str) const;

    operator const Vector<String>&() const { return m_strings; }

private:
    DOMStringList() = default;
    explicit DOMStringList(Vector<String>&& strings)
        : m_strings(WTFMove(strings))
    {
    }

    Vector<String> m_strings;
};

} // namespace WebCore

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 16, 2025.
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
class DOMSetAdapter;

class InternalsSetLike : public RefCounted<InternalsSetLike> {
public:
    static Ref<InternalsSetLike> create() { return adoptRef(*new InternalsSetLike); }

    void clearFromSetLike() { m_items.clear(); }
    bool addToSetLike(const String&);
    bool removeFromSetLike(const String& item) { return m_items.removeFirst(item); }
    void initializeSetLike(DOMSetAdapter&);

    const Vector<String>& items() const { return m_items; }

private:
    InternalsSetLike();
    Vector<String> m_items;
};

inline bool InternalsSetLike::addToSetLike(const String& value)
{
    bool hasValue = removeFromSetLike(value);
    m_items.append(value);
    return !hasValue;
}

} // namespace WebCore

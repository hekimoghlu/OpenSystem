/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 15, 2023.
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

#include "ExceptionOr.h"
#include "ScriptExecutionContext.h"
#include <variant>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class DOMURL;
class ScriptExecutionContext;

class URLSearchParams : public RefCounted<URLSearchParams> {
public:
    ~URLSearchParams();

    static ExceptionOr<Ref<URLSearchParams>> create(std::variant<Vector<Vector<String>>, Vector<KeyValuePair<String, String>>, String>&&);
    static Ref<URLSearchParams> create(const String& string, DOMURL* associatedURL)
    {
        return adoptRef(*new URLSearchParams(string, associatedURL));
    }

    size_t size() const { return m_pairs.size(); }

    void append(const String& name, const String& value);
    void remove(const String& name, const String& value = { });
    String get(const String& name) const;
    Vector<String> getAll(const String& name) const;
    bool has(const String& name, const String& value = { }) const;
    void set(const String& name, const String& value);
    String toString() const;
    void updateFromAssociatedURL();
    void sort();

    class Iterator {
    public:
        explicit Iterator(URLSearchParams&);
        std::optional<KeyValuePair<String, String>> next();

    private:
        Ref<URLSearchParams> m_target;
        size_t m_index { 0 };
    };
    Iterator createIterator(ScriptExecutionContext*) { return Iterator { *this }; }

private:
    const Vector<KeyValuePair<String, String>>& pairs() const { return m_pairs; }
    URLSearchParams(const String&, DOMURL*);
    URLSearchParams(const Vector<KeyValuePair<String, String>>&);
    void updateURL();

    WeakPtr<DOMURL> m_associatedURL;
    Vector<KeyValuePair<String, String>> m_pairs;
};

} // namespace WebCore

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 14, 2022.
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

#include "File.h"
#include <pal/text/TextEncoding.h>
#include <variant>
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

template<typename> class ExceptionOr;
class HTMLElement;
class HTMLFormElement;

class DOMFormData : public RefCounted<DOMFormData>, public ContextDestructionObserver {
public:
    using FormDataEntryValue = std::variant<RefPtr<File>, String>;

    struct Item {
        String name;
        FormDataEntryValue data;
    };

    static ExceptionOr<Ref<DOMFormData>> create(ScriptExecutionContext&, HTMLFormElement*, HTMLElement*);
    static Ref<DOMFormData> create(ScriptExecutionContext*, const PAL::TextEncoding&);

    const Vector<Item>& items() const { return m_items; }
    const PAL::TextEncoding& encoding() const { return m_encoding; }

    void append(const String& name, const String& value);
    void append(const String& name, Blob&, const String& filename = { });
    void remove(const String& name);
    std::optional<FormDataEntryValue> get(const String& name);
    Vector<FormDataEntryValue> getAll(const String& name);
    bool has(const String& name);
    void set(const String& name, const String& value);
    void set(const String& name, Blob&, const String& filename = { });
    Ref<DOMFormData> clone() const;

    class Iterator {
    public:
        explicit Iterator(DOMFormData&);
        std::optional<KeyValuePair<String, FormDataEntryValue>> next();

    private:
        Ref<DOMFormData> m_target;
        size_t m_index { 0 };
    };
    Iterator createIterator(ScriptExecutionContext*) { return Iterator { *this }; }

private:
    explicit DOMFormData(ScriptExecutionContext*, const PAL::TextEncoding& = PAL::UTF8Encoding());

    void set(const String& name, Item&&);

    PAL::TextEncoding m_encoding;
    Vector<Item> m_items;
};

} // namespace WebCore

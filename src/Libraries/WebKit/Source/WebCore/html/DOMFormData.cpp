/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 14, 2022.
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
#include "DOMFormData.h"

#include "Document.h"
#include "HTMLFormControlElement.h"
#include "HTMLFormElement.h"

namespace WebCore {

DOMFormData::DOMFormData(ScriptExecutionContext* context, const PAL::TextEncoding& encoding)
    : ContextDestructionObserver(context)
    , m_encoding(encoding)
{
}

ExceptionOr<Ref<DOMFormData>> DOMFormData::create(ScriptExecutionContext& context, HTMLFormElement* form, HTMLElement* submitter)
{
    // https://xhr.spec.whatwg.org/#dom-formdata
    auto formData = adoptRef(*new DOMFormData(&context));
    if (!form)
        return formData;

    RefPtr<HTMLFormControlElement> control;
    if (submitter) {
        control = dynamicDowncast<HTMLFormControlElement>(*submitter);
        if (!control || !control->isSubmitButton())
            return Exception { ExceptionCode::TypeError, "The specified element is not a submit button."_s };
        if (control->form() != form)
            return Exception { ExceptionCode::NotFoundError, "The specified element is not owned by this form element."_s };
    }
    auto result = form->constructEntryList(control.get(), WTFMove(formData), nullptr);
    
    if (!result)
        return Exception { ExceptionCode::InvalidStateError, "Already constructing Form entry list."_s };
    
    return result.releaseNonNull();
}

Ref<DOMFormData> DOMFormData::create(ScriptExecutionContext* context, const PAL::TextEncoding& encoding)
{
    return adoptRef(*new DOMFormData(context, encoding));
}

Ref<DOMFormData> DOMFormData::clone() const
{
    auto newFormData = adoptRef(*new DOMFormData(scriptExecutionContext(), this->encoding()));
    newFormData->m_items = m_items;
    
    return newFormData;
}

// https://html.spec.whatwg.org/multipage/form-control-infrastructure.html#create-an-entry
static auto createStringEntry(const String& name, const String& value) -> DOMFormData::Item
{
    return {
        replaceUnpairedSurrogatesWithReplacementCharacter(String(name)),
        replaceUnpairedSurrogatesWithReplacementCharacter(String(value)),
    };
}

// https://html.spec.whatwg.org/multipage/form-control-infrastructure.html#create-an-entry
static auto createFileEntry(const String& name, Blob& blob, const String& filename) -> DOMFormData::Item
{
    auto usvName = replaceUnpairedSurrogatesWithReplacementCharacter(String(name));

    if (RefPtr file = dynamicDowncast<File>(blob)) {
        if (!filename.isNull())
            return { usvName, File::create(blob.scriptExecutionContext(), *file, filename) };
        return { usvName, WTFMove(file) };
    }
    return { usvName, File::create(blob.scriptExecutionContext(), blob, filename.isNull() ? "blob"_s : filename) };
}

void DOMFormData::append(const String& name, const String& value)
{
    m_items.append(createStringEntry(name, value));
}

void DOMFormData::append(const String& name, Blob& blob, const String& filename)
{
    m_items.append(createFileEntry(name, blob, filename));
}

void DOMFormData::remove(const String& name)
{
    m_items.removeAllMatching([&name] (const auto& item) {
        return item.name == name;
    });
}

auto DOMFormData::get(const String& name) -> std::optional<FormDataEntryValue>
{
    for (auto& item : m_items) {
        if (item.name == name)
            return item.data;
    }

    return std::nullopt;
}

auto DOMFormData::getAll(const String& name) -> Vector<FormDataEntryValue>
{
    Vector<FormDataEntryValue> result;

    for (auto& item : m_items) {
        if (item.name == name)
            result.append(item.data);
    }

    return result;
}

bool DOMFormData::has(const String& name)
{
    for (auto& item : m_items) {
        if (item.name == name)
            return true;
    }
    
    return false;
}

void DOMFormData::set(const String& name, const String& value)
{
    set(name, { name, value });
}

void DOMFormData::set(const String& name, Blob& blob, const String& filename)
{
    set(name, createFileEntry(name, blob, filename));
}

void DOMFormData::set(const String& name, Item&& item)
{
    std::optional<size_t> initialMatchLocation;

    // Find location of the first item with a matching name.
    for (size_t i = 0; i < m_items.size(); ++i) {
        if (name == m_items[i].name) {
            initialMatchLocation = i;
            break;
        }
    }

    if (initialMatchLocation) {
        m_items[*initialMatchLocation] = WTFMove(item);

        m_items.removeAllMatching([&name] (const auto& item) {
            return item.name == name;
        }, *initialMatchLocation + 1);
        return;
    }

    m_items.append(WTFMove(item));
}

DOMFormData::Iterator::Iterator(DOMFormData& target)
    : m_target(target)
{
}

std::optional<KeyValuePair<String, DOMFormData::FormDataEntryValue>> DOMFormData::Iterator::next()
{
    auto& items = m_target->items();
    if (m_index >= items.size())
        return std::nullopt;

    auto& item = items[m_index++];
    return makeKeyValuePair(item.name, item.data);
}

} // namespace WebCore

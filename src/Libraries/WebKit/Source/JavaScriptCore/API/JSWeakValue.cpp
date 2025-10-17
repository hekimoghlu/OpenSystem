/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 17, 2023.
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
#include "JSWeakValue.h"

#include "JSCInlines.h"

namespace JSC {

JSWeakValue::~JSWeakValue()
{
    clear();
}

void JSWeakValue::clear()
{
    switch (m_tag) {
    case WeakTypeTag::NotSet:
        return;
    case WeakTypeTag::Primitive:
        m_value.primitive = JSValue();
        return;
    case WeakTypeTag::Object:
        m_value.object.clear();
        return;
    case WeakTypeTag::String:
        m_value.string.clear();
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

bool JSWeakValue::isClear() const
{
    switch (m_tag) {
    case WeakTypeTag::NotSet:
        return true;
    case WeakTypeTag::Primitive:
        return !m_value.primitive;
    case WeakTypeTag::Object:
        return !m_value.object;
    case WeakTypeTag::String:
        return !m_value.string;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

void JSWeakValue::setPrimitive(JSValue primitive)
{
    ASSERT(!isSet());
    ASSERT(!m_value.primitive);
    ASSERT(primitive.isPrimitive());
    m_tag = WeakTypeTag::Primitive;
    m_value.primitive = primitive;
}

void JSWeakValue::setObject(JSObject* object, WeakHandleOwner& owner, void* context)
{
    ASSERT(!isSet());
    ASSERT(!m_value.object);
    m_tag = WeakTypeTag::Object;
    Weak<JSObject> weak(object, &owner, context);
    m_value.object.swap(weak);
}

void JSWeakValue::setString(JSString* string, WeakHandleOwner& owner, void* context)
{
    ASSERT(!isSet());
    ASSERT(!m_value.string);
    m_tag = WeakTypeTag::String;
    Weak<JSString> weak(string, &owner, context);
    m_value.string.swap(weak);
}

} // namespace JSC

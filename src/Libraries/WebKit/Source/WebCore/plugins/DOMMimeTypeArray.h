/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 9, 2023.
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

#include "ScriptWrappable.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class DOMMimeType;
class Navigator;

class DOMMimeTypeArray final : public ScriptWrappable, public RefCounted<DOMMimeTypeArray> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DOMMimeTypeArray);
public:
    static Ref<DOMMimeTypeArray> create(Navigator&, Vector<Ref<DOMMimeType>>&& = { });
    ~DOMMimeTypeArray();

    unsigned length() const;
    bool isSupportedPropertyIndex(unsigned index) const { return index < m_types.size(); }
    RefPtr<DOMMimeType> item(unsigned index);
    RefPtr<DOMMimeType> namedItem(const AtomString& propertyName);
    Vector<AtomString> supportedPropertyNames() const;

    bool isSupportedPropertyName(const AtomString&) const;

    Navigator* navigator() { return m_navigator.get(); }

private:
    explicit DOMMimeTypeArray(Navigator&, Vector<Ref<DOMMimeType>>&&);
    
    WeakPtr<Navigator> m_navigator;
    Vector<Ref<DOMMimeType>> m_types;
};

} // namespace WebCore

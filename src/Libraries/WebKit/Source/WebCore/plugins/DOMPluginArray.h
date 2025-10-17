/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 21, 2022.
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

#include "Navigator.h"
#include "ScriptWrappable.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class DOMPlugin;
class Navigator;

class DOMPluginArray final : public ScriptWrappable, public RefCounted<DOMPluginArray> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DOMPluginArray);
public:
    static Ref<DOMPluginArray> create(Navigator&, Vector<Ref<DOMPlugin>>&& = { }, Vector<Ref<DOMPlugin>>&& = { });
    ~DOMPluginArray();

    unsigned length() const;
    bool isSupportedPropertyIndex(unsigned index) const { return index < length(); }
    RefPtr<DOMPlugin> item(unsigned index);
    RefPtr<DOMPlugin> namedItem(const AtomString& propertyName);
    Vector<AtomString> supportedPropertyNames() const;
    bool isSupportedPropertyName(const AtomString&) const;

    void refresh(bool reloadPages);

    Navigator* navigator() { return m_navigator.get(); }
    
private:
    explicit DOMPluginArray(Navigator&, Vector<Ref<DOMPlugin>>&&, Vector<Ref<DOMPlugin>>&&);


    WeakPtr<Navigator> m_navigator;
    Vector<Ref<DOMPlugin>> m_publiclyVisiblePlugins;
    Vector<Ref<DOMPlugin>> m_additionalWebVisibilePlugins;
};

} // namespace WebCore

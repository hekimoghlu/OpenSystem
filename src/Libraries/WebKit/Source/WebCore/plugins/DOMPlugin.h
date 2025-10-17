/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 29, 2025.
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

#include "PluginData.h"
#include "ScriptWrappable.h"
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/RefPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class DOMMimeType;
class Navigator;

class DOMPlugin final : public RefCountedAndCanMakeWeakPtr<DOMPlugin>, public ScriptWrappable {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DOMPlugin);
public:
    static Ref<DOMPlugin> create(Navigator&, const PluginInfo&);
    ~DOMPlugin();

    const PluginInfo& info() const { return m_info; }

    String name() const;
    String filename() const;
    String description() const;

    unsigned length() const;

    bool isSupportedPropertyIndex(unsigned index) const { return index < m_mimeTypes.size(); }
    RefPtr<DOMMimeType> item(unsigned index);
    RefPtr<DOMMimeType> namedItem(const AtomString& propertyName);
    Vector<AtomString> supportedPropertyNames() const;
    bool isSupportedPropertyName(const AtomString&) const;

    const Vector<Ref<DOMMimeType>>& mimeTypes() const { return m_mimeTypes; }

    Navigator* navigator() { return m_navigator.get(); }

private:
    DOMPlugin(Navigator&, const PluginInfo&);

    WeakPtr<Navigator> m_navigator;
    PluginInfo m_info;
    Vector<Ref<DOMMimeType>> m_mimeTypes;
};

} // namespace WebCore

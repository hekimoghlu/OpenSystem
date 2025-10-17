/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 27, 2022.
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
#include "DOMPluginArray.h"

#include "DOMPlugin.h"
#include "LocalFrame.h"
#include "Page.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DOMPluginArray);

Ref<DOMPluginArray> DOMPluginArray::create(Navigator& navigator, Vector<Ref<DOMPlugin>>&& publiclyVisiblePlugins, Vector<Ref<DOMPlugin>>&& additionalWebVisibilePlugins)
{
    return adoptRef(*new DOMPluginArray(navigator, WTFMove(publiclyVisiblePlugins), WTFMove(additionalWebVisibilePlugins)));
}

DOMPluginArray::DOMPluginArray(Navigator& navigator, Vector<Ref<DOMPlugin>>&& publiclyVisiblePlugins, Vector<Ref<DOMPlugin>>&& additionalWebVisibilePlugins)
    : m_navigator(navigator)
    , m_publiclyVisiblePlugins(WTFMove(publiclyVisiblePlugins))
    , m_additionalWebVisibilePlugins(WTFMove(additionalWebVisibilePlugins))
{
}

DOMPluginArray::~DOMPluginArray() = default;

unsigned DOMPluginArray::length() const
{
    return m_publiclyVisiblePlugins.size();
}

RefPtr<DOMPlugin> DOMPluginArray::item(unsigned index)
{
    if (index >= m_publiclyVisiblePlugins.size())
        return nullptr;
    return m_publiclyVisiblePlugins[index].ptr();
}

RefPtr<DOMPlugin> DOMPluginArray::namedItem(const AtomString& propertyName)
{
    for (auto& plugin : m_publiclyVisiblePlugins) {
        if (plugin->name() == propertyName)
            return plugin.ptr();
    }

    for (auto& plugin : m_additionalWebVisibilePlugins) {
        if (plugin->name() == propertyName)
            return plugin.ptr();
    }

    return nullptr;
}

bool DOMPluginArray::isSupportedPropertyName(const AtomString& propertyName) const
{
    return m_publiclyVisiblePlugins.containsIf([&](auto& plugin) { return plugin->name() == propertyName; })
        || m_additionalWebVisibilePlugins.containsIf([&](auto& plugin) { return plugin->name() == propertyName; });
}

Vector<AtomString> DOMPluginArray::supportedPropertyNames() const
{
    return m_publiclyVisiblePlugins.map([](auto& plugin) {
        return AtomString { plugin->name() };
    });
}

void DOMPluginArray::refresh(bool reloadPages)
{
    if (!m_navigator)
        return;

    auto* frame = m_navigator->frame();
    if (!frame)
        return;

    if (!frame->page())
        return;

    Page::refreshPlugins(reloadPages);
}

} // namespace WebCore

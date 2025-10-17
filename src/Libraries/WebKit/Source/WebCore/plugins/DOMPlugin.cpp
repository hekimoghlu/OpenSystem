/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 21, 2024.
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
#include "DOMPlugin.h"

#include "DOMMimeType.h"
#include "Navigator.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DOMPlugin);

Ref<DOMPlugin> DOMPlugin::create(Navigator& navigator, const PluginInfo& info)
{
    return adoptRef(*new DOMPlugin(navigator, info));
}

static Vector<Ref<DOMMimeType>> makeMimeTypes(Navigator& navigator, const PluginInfo& info, DOMPlugin& self)
{
    auto types = info.mimes.map([&](auto& type) {
        return DOMMimeType::create(navigator, type, self);
    });
    std::sort(types.begin(), types.end(), [](const Ref<DOMMimeType>& a, const Ref<DOMMimeType>& b) {
        return codePointCompareLessThan(a->type(), b->type());
    });

    return types;
}

DOMPlugin::DOMPlugin(Navigator& navigator, const PluginInfo& info)
    : m_navigator(navigator)
    , m_info(info)
    , m_mimeTypes(makeMimeTypes(navigator, info, *this))
{
}

DOMPlugin::~DOMPlugin() = default;

String DOMPlugin::name() const
{
    return m_info.name;
}

String DOMPlugin::filename() const
{
    return m_info.file;
}

String DOMPlugin::description() const
{
    return m_info.desc;
}

unsigned DOMPlugin::length() const
{
    return m_mimeTypes.size();
}

RefPtr<DOMMimeType> DOMPlugin::item(unsigned index)
{
    if (index >= m_mimeTypes.size())
        return nullptr;
    return m_mimeTypes[index].ptr();
}

RefPtr<DOMMimeType> DOMPlugin::namedItem(const AtomString& propertyName)
{
    for (auto& type : m_mimeTypes) {
        if (type->type() == propertyName)
            return type.ptr();
    }
    return nullptr;
}

bool DOMPlugin::isSupportedPropertyName(const AtomString& propertyName) const
{
    return m_mimeTypes.containsIf([&](auto& type) { return type->type() == propertyName; });
}

Vector<AtomString> DOMPlugin::supportedPropertyNames() const
{
    return m_mimeTypes.map([](auto& type) {
        return type->type();
    });
}

} // namespace WebCore

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 30, 2022.
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
#include "DOMMimeType.h"

#include "DOMPlugin.h"
#include "Navigator.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {

Ref<DOMMimeType> DOMMimeType::create(Navigator& navigator, const MimeClassInfo& info, DOMPlugin& enabledPlugin)
{
    return adoptRef(*new DOMMimeType(navigator, info, enabledPlugin));
}

DOMMimeType::DOMMimeType(Navigator& navigator, const MimeClassInfo& info, DOMPlugin& enabledPlugin)
    : m_navigator(navigator)
    , m_info(info)
    , m_enabledPlugin(enabledPlugin)
{
}

DOMMimeType::~DOMMimeType() = default;

AtomString DOMMimeType::type() const
{
    return m_info.type;
}

String DOMMimeType::suffixes() const
{
    StringBuilder builder;
    for (size_t i = 0; i < m_info.extensions.size(); ++i) {
        if (i)
            builder.append(',');
        builder.append(m_info.extensions[i]);
    }
    return builder.toString();
}

String DOMMimeType::description() const
{
    return m_info.desc;
}

RefPtr<DOMPlugin> DOMMimeType::enabledPlugin() const
{
    return m_enabledPlugin.get();
}

} // namespace WebCore

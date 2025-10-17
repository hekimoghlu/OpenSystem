/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 19, 2023.
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
#include "WebColorPicker.h"

namespace WebKit {

WebColorPicker::WebColorPicker(Client* client)
    : m_client(client)
{
}

WebColorPicker::~WebColorPicker()
{
}

void WebColorPicker::endPicker()
{
    if (CheckedPtr client = std::exchange(m_client, nullptr))
        client->didEndColorPicker();
}

void WebColorPicker::setSelectedColor(const WebCore::Color& color)
{
    if (CheckedPtr client = m_client)
        client->didChooseColor(color);
}

void WebColorPicker::showColorPicker(const WebCore::Color&)
{
    ASSERT_NOT_REACHED();
    return;
}

} // namespace WebKit

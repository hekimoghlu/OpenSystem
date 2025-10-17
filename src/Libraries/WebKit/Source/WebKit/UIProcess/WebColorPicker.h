/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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

#include <wtf/CheckedPtr.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WebCore {
class Color;
}

namespace WebKit {

class WebPageProxy;

class WebColorPickerClient : public CanMakeCheckedPtr<WebColorPickerClient> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WebColorPickerClient);
public:
    virtual void didChooseColor(const WebCore::Color&) = 0;
    virtual void didEndColorPicker() = 0;

protected:
    virtual ~WebColorPickerClient() = default;
};

class WebColorPicker : public RefCounted<WebColorPicker> {
public:
    using Client = WebColorPickerClient;

    static Ref<WebColorPicker> create(Client* client)
    {
        return adoptRef(*new WebColorPicker(client));
    }

    virtual ~WebColorPicker();

    virtual void endPicker();
    virtual void setSelectedColor(const WebCore::Color&);
    virtual void showColorPicker(const WebCore::Color&);

protected:
    explicit WebColorPicker(Client*);

    Client* client() const { return m_client.get(); }

private:
    CheckedPtr<Client> m_client;
};

} // namespace WebKit

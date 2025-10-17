/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 17, 2025.
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

#include "APIObject.h"
#include <wtf/Forward.h>
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace API {
class Array;
class Data;
}

namespace WebKit {

class WebPageProxy;
class WebProcessProxy;

class WebOpenPanelResultListenerProxy : public API::ObjectImpl<API::Object::Type::FramePolicyListener> {
public:
    static Ref<WebOpenPanelResultListenerProxy> create(WebPageProxy* page, WebProcessProxy& process)
    {
        return adoptRef(*new WebOpenPanelResultListenerProxy(page, process));
    }

    virtual ~WebOpenPanelResultListenerProxy();

#if PLATFORM(IOS_FAMILY)
    void chooseFiles(const Vector<String>& filenames, const String& displayString, const API::Data* iconImageData);
#endif
    void chooseFiles(const Vector<String>& filenames, const Vector<String>& allowedMIMETypes = { });
    void cancel();

    void invalidate();

    WebProcessProxy* process() const;

private:
    WebOpenPanelResultListenerProxy(WebPageProxy*, WebProcessProxy&);

    RefPtr<WebPageProxy> m_page;
    WeakPtr<WebProcessProxy> m_process;
};

} // namespace WebKit

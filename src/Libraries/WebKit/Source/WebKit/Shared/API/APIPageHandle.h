/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 27, 2022.
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
#include "WebPageProxyIdentifier.h"
#include <WebCore/PageIdentifier.h>
#include <wtf/Ref.h>

namespace API {

class PageHandle : public ObjectImpl<Object::Type::PageHandle> {
public:
    static Ref<PageHandle> create(WebKit::WebPageProxyIdentifier pageProxyID, WebCore::PageIdentifier webPageID)
    {
        return adoptRef(*new PageHandle(pageProxyID, webPageID, false));
    }
    static Ref<PageHandle> createAutoconverting(WebKit::WebPageProxyIdentifier pageProxyID, WebCore::PageIdentifier webPageID)
    {
        return adoptRef(*new PageHandle(pageProxyID, webPageID, true));
    }
    static Ref<PageHandle> create(WebKit::WebPageProxyIdentifier pageProxyID, WebCore::PageIdentifier webPageID, bool autoconverting)
    {
        return adoptRef(*new PageHandle(pageProxyID, webPageID, autoconverting));
    }

    virtual ~PageHandle() = default;

    WebKit::WebPageProxyIdentifier pageProxyID() const { return m_pageProxyID; }
    WebCore::PageIdentifier webPageID() const { return m_webPageID; }
    bool isAutoconverting() const { return m_isAutoconverting; }

private:
    PageHandle(WebKit::WebPageProxyIdentifier pageProxyID, WebCore::PageIdentifier webPageID, bool isAutoconverting)
        : m_pageProxyID(pageProxyID)
        , m_webPageID(webPageID)
        , m_isAutoconverting(isAutoconverting)
    {
    }

    const WebKit::WebPageProxyIdentifier m_pageProxyID;
    const WebCore::PageIdentifier m_webPageID;
    const bool m_isAutoconverting;
};

} // namespace API

SPECIALIZE_TYPE_TRAITS_API_OBJECT(PageHandle);

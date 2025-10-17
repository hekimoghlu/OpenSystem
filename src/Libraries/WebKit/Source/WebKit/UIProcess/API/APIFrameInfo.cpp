/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 22, 2024.
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
#include "APIFrameInfo.h"

#include "APIFrameHandle.h"
#include "FrameInfoData.h"
#include "WebFrameProxy.h"
#include "WebPageProxy.h"
#include <utility>

namespace API {

Ref<FrameInfo> FrameInfo::create(WebKit::FrameInfoData&& frameInfoData, RefPtr<WebKit::WebPageProxy>&& page)
{
    return adoptRef(*new FrameInfo(WTFMove(frameInfoData), std::forward<RefPtr<WebKit::WebPageProxy>&&>(page)));
}

FrameInfo::FrameInfo(WebKit::FrameInfoData&& data, RefPtr<WebKit::WebPageProxy>&& page)
    : m_data(WTFMove(data))
    , m_page(WTFMove(page)) { }

FrameInfo::~FrameInfo() = default;

Ref<FrameHandle> FrameInfo::handle() const
{
    return FrameHandle::create(m_data.frameID);
}

RefPtr<FrameHandle> FrameInfo::parentFrameHandle() const
{
    if (!m_data.parentFrameID)
        return nullptr;
    return FrameHandle::create(*m_data.parentFrameID);
}

WTF::String FrameInfo::title() const
{
    if (!m_page)
        return { };

    if (RefPtr frame = WebKit::WebFrameProxy::webFrame(m_data.frameID); frame && frame->page() == m_page)
        return frame->title();

    return { };
}

} // namespace API

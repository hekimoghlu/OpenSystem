/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 5, 2024.
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
#include "FrameTreeNodeData.h"

namespace WebKit {
class WebPageProxy;
}

namespace API {

class FrameHandle;

class FrameTreeNode final : public ObjectImpl<Object::Type::FrameTreeNode> {
public:
    static Ref<FrameTreeNode> create(WebKit::FrameTreeNodeData&& data, WebKit::WebPageProxy& page) { return adoptRef(*new FrameTreeNode(WTFMove(data), page)); }
    virtual ~FrameTreeNode();

    WebKit::WebPageProxy& page() { return m_page.get(); }
    const WebKit::FrameInfoData& info() const { return m_data.info; }
    const Vector<WebKit::FrameTreeNodeData>& childFrames() const { return m_data.children; }

private:
    FrameTreeNode(WebKit::FrameTreeNodeData&& data, WebKit::WebPageProxy& page)
        : m_data(WTFMove(data))
        , m_page(page) { }

    const WebKit::FrameTreeNodeData m_data;
    Ref<WebKit::WebPageProxy> m_page;
};

} // namespace API

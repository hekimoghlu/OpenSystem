/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 16, 2024.
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
#include "FrameInfoData.h"
#include <WebCore/ResourceRequest.h>

namespace WebCore {
class SecurityOriginData;
}

namespace WebKit {
class WebFrameProxy;
class WebPageProxy;
struct FrameInfoData;
}

namespace API {

class FrameHandle;
class SecurityOrigin;

class FrameInfo final : public ObjectImpl<Object::Type::FrameInfo> {
public:
    static Ref<FrameInfo> create(WebKit::FrameInfoData&&, RefPtr<WebKit::WebPageProxy>&&);
    virtual ~FrameInfo();

    bool isMainFrame() const { return m_data.isMainFrame; }
    bool isLocalFrame() const { return m_data.frameType == WebKit::FrameType::Local; }
    const WebCore::ResourceRequest& request() const { return m_data.request; }
    WebCore::SecurityOriginData& securityOrigin() { return m_data.securityOrigin; }
    Ref<FrameHandle> handle() const;
    WebKit::WebPageProxy* page() { return m_page.get(); }
    RefPtr<FrameHandle> parentFrameHandle() const;
    Markable<WebCore::ScriptExecutionContextIdentifier> documentID() const { return m_data.documentID; }
    ProcessID processID() const { return m_data.processID; }
    bool isFocused() const { return m_data.isFocused; }
    bool errorOccurred() const { return m_data.errorOccurred; }
    WTF::String title() const;

    const WebKit::FrameInfoData& frameInfoData() const { return m_data; }

private:
    FrameInfo(WebKit::FrameInfoData&&, RefPtr<WebKit::WebPageProxy>&&);

    WebKit::FrameInfoData m_data;
    RefPtr<WebKit::WebPageProxy> m_page;
};

} // namespace API

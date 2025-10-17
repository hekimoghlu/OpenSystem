/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 19, 2022.
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

#include "APIFrameInfo.h"
#include "APIObject.h"
#include <WebCore/ResourceRequest.h>
#include <WebCore/ResourceResponse.h>

namespace API {

class FrameInfo;
class Navigation;

class NavigationResponse final : public ObjectImpl<Object::Type::NavigationResponse> {
public:
    template<typename... Args> static Ref<NavigationResponse> create(Args&&... args)
    {
        return adoptRef(*new NavigationResponse(std::forward<Args>(args)...));
    }
    ~NavigationResponse();

    FrameInfo& frame() { return m_frame.get(); }
    Ref<FrameInfo> protectedFrame() { return m_frame.get(); }

    const WebCore::ResourceRequest& request() const { return m_request; }
    const WebCore::ResourceResponse& response() const { return m_response; }

    bool canShowMIMEType() const { return m_canShowMIMEType; }
    const WTF::String& downloadAttribute() const { return m_downloadAttribute; }

    FrameInfo* navigationInitiatingFrame();

private:
    NavigationResponse(API::FrameInfo&, const WebCore::ResourceRequest&, const WebCore::ResourceResponse&, bool canShowMIMEType, const WTF::String& downloadAttribute, Navigation*);

    const Ref<FrameInfo> m_frame;
    WebCore::ResourceRequest m_request;
    WebCore::ResourceResponse m_response;
    bool m_canShowMIMEType;
    WTF::String m_downloadAttribute;
    const RefPtr<Navigation> m_navigation;
    RefPtr<FrameInfo> m_sourceFrame;
};

} // namespace API

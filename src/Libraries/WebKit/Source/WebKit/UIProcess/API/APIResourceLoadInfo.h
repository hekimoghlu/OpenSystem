/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 3, 2022.
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
#include "ResourceLoadInfo.h"

namespace API {

class ResourceLoadInfo final : public ObjectImpl<Object::Type::ResourceLoadInfo> {
public:
    static Ref<ResourceLoadInfo> create(WebKit::ResourceLoadInfo&& info)
    {
        return adoptRef(*new ResourceLoadInfo(WTFMove(info)));
    }

    explicit ResourceLoadInfo(WebKit::ResourceLoadInfo&& info)
        : m_info(WTFMove(info)) { }

    WebKit::NetworkResourceLoadIdentifier resourceLoadID() const { return m_info.resourceLoadID; }
    std::optional<WebCore::FrameIdentifier> frameID() const { return m_info.frameID; }
    std::optional<WebCore::FrameIdentifier> parentFrameID() const { return m_info.parentFrameID; }
    Markable<WTF::UUID> documentID() const { return m_info.documentID; }
    const WTF::URL& originalURL() const { return m_info.originalURL; }
    const WTF::String& originalHTTPMethod() const { return m_info.originalHTTPMethod; }
    WallTime eventTimestamp() const { return m_info.eventTimestamp; }
    bool loadedFromCache() const { return m_info.loadedFromCache; }
    WebKit::ResourceLoadInfo::Type resourceLoadType() const { return m_info.type; }

private:
    const WebKit::ResourceLoadInfo m_info;
};

} // namespace API

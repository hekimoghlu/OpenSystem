/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 16, 2023.
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
#include <WebCore/ElementTargetingTypes.h>

namespace WebKit {
class WebPageProxy;
}

namespace API {

class TargetedElementRequest final : public ObjectImpl<Object::Type::TargetedElementRequest> {
public:
    bool shouldIgnorePointerEventsNone() const { return m_request.shouldIgnorePointerEventsNone; }
    void setShouldIgnorePointerEventsNone(bool value) { m_request.shouldIgnorePointerEventsNone = value; }

    bool canIncludeNearbyElements() const { return m_request.canIncludeNearbyElements; }
    void setCanIncludeNearbyElements(bool value) { m_request.canIncludeNearbyElements = value; }

    WebCore::TargetedElementRequest makeRequest(const WebKit::WebPageProxy&) const;

    WebCore::FloatPoint point() const;
    void setPoint(WebCore::FloatPoint);

    void setSearchText(WTF::String&&);
    void setSelectors(WebCore::TargetedElementSelectors&&);

private:
    WebCore::TargetedElementRequest m_request;
};


} // namespace API

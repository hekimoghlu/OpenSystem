/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 20, 2025.
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

#if PLATFORM(IOS_FAMILY)

OBJC_CLASS NSArray;
OBJC_CLASS NSDictionary;

#include "APIObject.h"
#include "InteractionInformationAtPosition.h"
#include <wtf/RetainPtr.h>

namespace API {

class ContextMenuElementInfo final : public ObjectImpl<Object::Type::ContextMenuElementInfo> {
public:
    template<typename... Args> static Ref<ContextMenuElementInfo> create(Args&&... args)
    {
        return adoptRef(*new ContextMenuElementInfo(std::forward<Args>(args)...));
    }
    
    const WTF::URL& url() const { return m_interactionInformation.url; }

    const WebKit::InteractionInformationAtPosition& interactionInformation() const { return m_interactionInformation; }

    const RetainPtr<NSDictionary> userInfo() const { return m_userInfo; }

private:
    ContextMenuElementInfo(const WebKit::InteractionInformationAtPosition&);
    ContextMenuElementInfo(const WebKit::InteractionInformationAtPosition&, NSDictionary *);
    
    WebKit::InteractionInformationAtPosition m_interactionInformation;
    RetainPtr<NSDictionary> m_userInfo;
};

} // namespace API

#endif

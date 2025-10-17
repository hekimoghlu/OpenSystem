/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 29, 2025.
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
#include "IdentifierTypes.h"
#include "WebPageGroupData.h"
#include "WebProcessProxy.h"
#include <WebCore/UserStyleSheetTypes.h>
#include <wtf/CheckedRef.h>
#include <wtf/WeakHashSet.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

class WebPreferences;
class WebPageProxy;

class WebPageGroup : public API::ObjectImpl<API::Object::Type::PageGroup>, public CanMakeWeakPtr<WebPageGroup> {
public:
    explicit WebPageGroup(const String& identifier = { });
    static Ref<WebPageGroup> create(const String& identifier = { });

    virtual ~WebPageGroup();

    PageGroupIdentifier pageGroupID() const { return m_data.pageGroupID; }

    const WebPageGroupData& data() const { return m_data; }

    WebPreferences& preferences() const;
    Ref<WebPreferences> protectedPreferences() const;

private:
    WebPageGroupData m_data;
    Ref<WebPreferences> m_preferences;
};

} // namespace WebKit

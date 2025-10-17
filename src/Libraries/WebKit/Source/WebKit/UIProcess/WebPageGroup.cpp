/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 14, 2025.
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
#include "WebPageGroup.h"

#include "APIArray.h"
#include "APIContentRuleList.h"
#include "APIUserScript.h"
#include "APIUserStyleSheet.h"
#include "WebCompiledContentRuleList.h"
#include "WebPageProxy.h"
#include "WebPreferences.h"
#include <wtf/CheckedPtr.h>
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/text/MakeString.h>

namespace WebKit {

using WebPageGroupMap = HashMap<PageGroupIdentifier, WeakRef<WebPageGroup>>;

static WebPageGroupMap& webPageGroupMap()
{
    static NeverDestroyed<WebPageGroupMap> map;
    return map;
}

Ref<WebPageGroup> WebPageGroup::create(const String& identifier)
{
    return adoptRef(*new WebPageGroup(identifier));
}

static WebPageGroupData pageGroupData(const String& identifier)
{
    static NeverDestroyed<HashMap<String, PageGroupIdentifier>> map;
    auto pageGroupID = [&] {
        if (HashMap<String, PageGroupIdentifier>::isValidKey(identifier)) {
            return map.get().ensure(identifier, [] {
                return PageGroupIdentifier::generate();
            }).iterator->value;
        }
        return PageGroupIdentifier::generate();
    }();

    String validIdentifier;
    if (!identifier.isEmpty())
        validIdentifier = identifier;
    else
        validIdentifier = makeString("__uniquePageGroupID-"_s, pageGroupID.toUInt64());

    return {
        WTFMove(validIdentifier),
        pageGroupID
    };
}

// FIXME: Why does the WebPreferences object here use ".WebKit2" instead of "WebKit2." which all the other constructors use.
// If it turns out that it's wrong, we can change it to to "WebKit2." and get rid of the globalDebugKeyPrefix from WebPreferences.
WebPageGroup::WebPageGroup(const String& identifier)
    : m_data(pageGroupData(identifier))
    , m_preferences(WebPreferences::createWithLegacyDefaults(m_data.identifier, ".WebKit2"_s, "WebKit2."_s))
{
    webPageGroupMap().set(m_data.pageGroupID, *this);
}

WebPageGroup::~WebPageGroup()
{
    webPageGroupMap().remove(pageGroupID());
}

WebPreferences& WebPageGroup::preferences() const
{
    return m_preferences;
}

Ref<WebPreferences> WebPageGroup::protectedPreferences() const
{
    return m_preferences;
}

} // namespace WebKit

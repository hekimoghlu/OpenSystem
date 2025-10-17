/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 27, 2024.
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
#include "APIContentRuleListAction.h"

#if ENABLE(CONTENT_EXTENSIONS)

namespace API {

Ref<ContentRuleListAction> ContentRuleListAction::create(WebCore::ContentRuleListResults::Result&& result)
{
    return adoptRef(*new ContentRuleListAction(WTFMove(result)));
}

ContentRuleListAction::ContentRuleListAction(WebCore::ContentRuleListResults::Result&& result)
    : m_result(WTFMove(result))
{
}

ContentRuleListAction::~ContentRuleListAction() = default;

bool ContentRuleListAction::blockedLoad() const
{
    return m_result.blockedLoad;
}

bool ContentRuleListAction::madeHTTPS() const
{
    return m_result.madeHTTPS;
}

bool ContentRuleListAction::blockedCookies() const
{
    return m_result.blockedCookies;
}

bool ContentRuleListAction::redirected() const
{
    return m_result.redirected;
}

bool ContentRuleListAction::modifiedHeaders() const
{
    return m_result.modifiedHeaders;
}

const Vector<WTF::String>& ContentRuleListAction::notifications() const
{
    return m_result.notifications;
}

} // namespace API

#endif // ENABLE(CONTENT_EXTENSIONS)

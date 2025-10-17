/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 18, 2023.
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

#include "WebPage.h"

#include "WebPageProxyIdentifier.h"
#include "WebUserContentController.h"
#include <WebCore/ActivityState.h>
#include <WebCore/IntPoint.h>
#include <WebCore/IntRect.h>
#include <WebCore/Page.h>

namespace WebKit {

inline WebCore::IntRect WebPage::bounds() const
{
    return WebCore::IntRect(WebCore::IntPoint(), size());
}

inline StorageNamespaceIdentifier WebPage::sessionStorageNamespaceIdentifier() const
{
    return ObjectIdentifier<StorageNamespaceIdentifierType>(m_webPageProxyIdentifier.toUInt64());
}

inline void WebPage::setHiddenPageDOMTimerThrottlingIncreaseLimit(Seconds limit)
{
    m_page->setDOMTimerAlignmentIntervalIncreaseLimit(limit);
}

inline bool WebPage::isVisible() const
{
    return m_activityState.contains(WebCore::ActivityState::IsVisible);
}

inline bool WebPage::isVisibleOrOccluded() const
{
    return m_activityState.contains(WebCore::ActivityState::IsVisibleOrOccluded);
}

inline UserContentControllerIdentifier WebPage::userContentControllerIdentifier() const
{
    return m_userContentController->identifier();
}

} // namespace WebKit

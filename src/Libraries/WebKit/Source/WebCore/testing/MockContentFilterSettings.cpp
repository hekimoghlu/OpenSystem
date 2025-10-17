/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 31, 2023.
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
#include "MockContentFilterSettings.h"

#if ENABLE(CONTENT_FILTERING)

#include "ContentFilter.h"
#include "ContentFilterUnblockHandler.h"
#include "MockContentFilter.h"
#include "MockContentFilterManager.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

MockContentFilterSettings& MockContentFilterSettings::singleton()
{
    MockContentFilter::ensureInstalled();
    static NeverDestroyed<MockContentFilterSettings> settings;
    return settings;
}

void MockContentFilterSettings::reset()
{
    singleton() = MockContentFilterSettings();
    MockContentFilterManager::singleton().notifySettingsChanged(singleton());
}

void MockContentFilterSettings::setEnabled(bool enabled)
{
    m_enabled = enabled;
    MockContentFilterManager::singleton().notifySettingsChanged(singleton());
}

void MockContentFilterSettings::setBlockedString(const String& blockedString)
{
    m_blockedString = blockedString;
    MockContentFilterManager::singleton().notifySettingsChanged(singleton());
}

void MockContentFilterSettings::setDecisionPoint(DecisionPoint decisionPoint)
{
    m_decisionPoint = decisionPoint;
    MockContentFilterManager::singleton().notifySettingsChanged(singleton());
}

void MockContentFilterSettings::setDecision(Decision decision)
{
    m_decision = decision;
    MockContentFilterManager::singleton().notifySettingsChanged(singleton());
}

void MockContentFilterSettings::setUnblockRequestDecision(Decision unblockRequestDecision)
{
    m_unblockRequestDecision = unblockRequestDecision;
    MockContentFilterManager::singleton().notifySettingsChanged(singleton());
}

void MockContentFilterSettings::setModifiedRequestURL(const String& modifiedRequestURL)
{
    m_modifiedRequestURL = modifiedRequestURL;
    MockContentFilterManager::singleton().notifySettingsChanged(singleton());
}

const String& MockContentFilterSettings::unblockRequestURL() const
{
    static NeverDestroyed<String> unblockRequestURL = makeString(ContentFilter::urlScheme(), "://"_s, unblockURLHost());
    return unblockRequestURL;
}

}; // namespace WebCore

#endif // ENABLE(CONTENT_FILTERING)

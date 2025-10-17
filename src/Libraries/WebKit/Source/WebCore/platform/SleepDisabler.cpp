/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 4, 2025.
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
#include "SleepDisabler.h"

#include "SleepDisablerClient.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SleepDisabler);

SleepDisabler::SleepDisabler(const String& reason, PAL::SleepDisabler::Type type, std::optional<PageIdentifier> pageID)
    : m_type(type)
    , m_pageID(pageID)
{
    if (sleepDisablerClient()) {
        m_identifier = SleepDisablerIdentifier::generate();
        sleepDisablerClient()->didCreateSleepDisabler(*m_identifier, reason, type == PAL::SleepDisabler::Type::Display, pageID);
        return;
    }

    m_platformSleepDisabler = PAL::SleepDisabler::create(reason, type);
}

SleepDisabler::~SleepDisabler()
{
    if (sleepDisablerClient())
        sleepDisablerClient()->didDestroySleepDisabler(*m_identifier, m_pageID);
}

} // namespace WebCore

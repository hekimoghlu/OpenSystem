/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 11, 2024.
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
#include "PageHeapAgent.h"
#include <wtf/TZoneMallocInlines.h>


namespace WebCore {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(PageHeapAgent);

PageHeapAgent::PageHeapAgent(PageAgentContext& context)
    : WebHeapAgent(context)
    , m_instrumentingAgents(context.instrumentingAgents)
{
}

PageHeapAgent::~PageHeapAgent() = default;

Inspector::Protocol::ErrorStringOr<void> PageHeapAgent::enable()
{
    auto result = WebHeapAgent::enable();

    m_instrumentingAgents.setEnabledPageHeapAgent(this);

    return result;
}

Inspector::Protocol::ErrorStringOr<void> PageHeapAgent::disable()
{
    m_instrumentingAgents.setEnabledPageHeapAgent(nullptr);

    return WebHeapAgent::disable();
}

void PageHeapAgent::mainFrameNavigated()
{
    clearHeapSnapshots();
}

} // namespace WebCore

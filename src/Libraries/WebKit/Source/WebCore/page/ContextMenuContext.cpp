/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 10, 2023.
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
#include "ContextMenuContext.h"

#include "Event.h"

#if ENABLE(CONTEXT_MENUS)

namespace WebCore {

ContextMenuContext::ContextMenuContext() = default;
ContextMenuContext::~ContextMenuContext() = default;

ContextMenuContext& ContextMenuContext::operator=(const ContextMenuContext&) = default;

ContextMenuContext::ContextMenuContext(Type type, const HitTestResult& hitTestResult, RefPtr<Event>&& event)
    : m_type(type)
    , m_hitTestResult(hitTestResult)
    , m_event(WTFMove(event))
    , m_hasEntireImage(hitTestResult.hasEntireImage())
{
}

} // namespace WebCore

#endif // ENABLE(CONTEXT_MENUS)

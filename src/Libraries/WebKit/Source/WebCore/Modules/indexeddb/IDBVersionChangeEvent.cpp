/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 4, 2025.
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
#include "IDBVersionChangeEvent.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(IDBVersionChangeEvent);

IDBVersionChangeEvent::IDBVersionChangeEvent(std::optional<IDBResourceIdentifier> requestIdentifier, uint64_t oldVersion, uint64_t newVersion, const AtomString& name)
    : Event(EventInterfaceType::IDBVersionChangeEvent, name, CanBubble::No, IsCancelable::No)
    , m_requestIdentifier(requestIdentifier)
    , m_oldVersion(oldVersion)
{
    if (newVersion)
        m_newVersion = newVersion;
    else
        m_newVersion = std::nullopt;
}

IDBVersionChangeEvent::IDBVersionChangeEvent(const AtomString& name, const Init& init, IsTrusted isTrusted)
    : Event(EventInterfaceType::IDBVersionChangeEvent, name, init, isTrusted)
    , m_oldVersion(init.oldVersion)
    , m_newVersion(init.newVersion)
{
}

} // namespace WebCore

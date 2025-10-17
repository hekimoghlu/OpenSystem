/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 8, 2024.
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

#include "Event.h"
#include "EventInit.h"
#include "NavigationHistoryEntry.h"
#include "NavigationNavigationType.h"

namespace WebCore {

class NavigationCurrentEntryChangeEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(NavigationCurrentEntryChangeEvent);
public:
    struct Init : EventInit {
        std::optional<NavigationNavigationType> navigationType;
        RefPtr<NavigationHistoryEntry> from;
    };

    static Ref<NavigationCurrentEntryChangeEvent> create(const AtomString& type, const Init&);

    std::optional<NavigationNavigationType> navigationType() const { return m_navigationType; };
    RefPtr<NavigationHistoryEntry> from() const { return m_from; };

private:
    NavigationCurrentEntryChangeEvent(const AtomString& type, const Init&);

    std::optional<NavigationNavigationType> m_navigationType;
    RefPtr<NavigationHistoryEntry> m_from;
};

} // namespace WebCore

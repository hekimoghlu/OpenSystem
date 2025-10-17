/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 10, 2022.
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
#include "IDBResourceIdentifier.h"

namespace WebCore {

class IDBVersionChangeEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(IDBVersionChangeEvent);
public:
    static Ref<IDBVersionChangeEvent> create(uint64_t oldVersion, uint64_t newVersion, const AtomString& eventType)
    {
        return adoptRef(*new IDBVersionChangeEvent(std::nullopt, oldVersion, newVersion, eventType));
    }

    static Ref<IDBVersionChangeEvent> create(const IDBResourceIdentifier& requestIdentifier, uint64_t oldVersion, uint64_t newVersion, const AtomString& eventType)
    {
        return adoptRef(*new IDBVersionChangeEvent(requestIdentifier, oldVersion, newVersion, eventType));
    }

    struct Init : EventInit {
        uint64_t oldVersion { 0 };
        std::optional<uint64_t> newVersion;
    };

    static Ref<IDBVersionChangeEvent> create(const AtomString& type, const Init& initializer, IsTrusted isTrusted = IsTrusted::No)
    {
        return adoptRef(*new IDBVersionChangeEvent(type, initializer, isTrusted));
    }

    std::optional<IDBResourceIdentifier> requestIdentifier() const { return m_requestIdentifier; }

    bool isVersionChangeEvent() const final { return true; }

    uint64_t oldVersion() const { return m_oldVersion; }
    std::optional<uint64_t> newVersion() const { return m_newVersion; }

private:
    IDBVersionChangeEvent(std::optional<IDBResourceIdentifier> requestIdentifier, uint64_t oldVersion, uint64_t newVersion, const AtomString& eventType);
    IDBVersionChangeEvent(const AtomString&, const Init&, IsTrusted);

    std::optional<IDBResourceIdentifier> m_requestIdentifier;
    uint64_t m_oldVersion;
    std::optional<uint64_t> m_newVersion;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::IDBVersionChangeEvent)
    static bool isType(const WebCore::Event& event) { return event.isVersionChangeEvent(); }
SPECIALIZE_TYPE_TRAITS_END()

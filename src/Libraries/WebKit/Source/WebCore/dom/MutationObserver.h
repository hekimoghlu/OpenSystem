/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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

#include "ExceptionOr.h"
#include "GCReachableRef.h"
#include <wtf/Forward.h>
#include <wtf/HashSet.h>
#include <wtf/OptionSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakHashSet.h>

namespace JSC {
class AbstractSlotVisitor;
}

namespace WebCore {

class Document;
class HTMLSlotElement;
class MutationCallback;
class MutationObserverRegistration;
class MutationRecord;
class Node;
class WindowEventLoop;

enum class MutationObserverOptionType : uint8_t {
    // MutationType
    ChildList = 1 << 0,
    Attributes = 1 << 1,
    CharacterData = 1 << 2,

    // ObservationFlags
    Subtree = 1 << 3,
    AttributeFilter = 1 << 4,

    // DeliveryFlags
    AttributeOldValue = 1 << 5,
    CharacterDataOldValue = 1 << 6,
};

using MutationObserverOptions = OptionSet<MutationObserverOptionType>;
using MutationRecordDeliveryOptions = OptionSet<MutationObserverOptionType>;

class MutationObserver final : public RefCounted<MutationObserver> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MutationObserver);
public:
    static Ref<MutationObserver> create(Ref<MutationCallback>&&);

    ~MutationObserver();

    struct Init {
        bool childList;
        std::optional<bool> attributes;
        std::optional<bool> characterData;
        bool subtree;
        std::optional<bool> attributeOldValue;
        std::optional<bool> characterDataOldValue;
        std::optional<Vector<AtomString>> attributeFilter;
    };

    ExceptionOr<void> observe(Node&, const Init&);
    
    struct TakenRecords {
        Vector<Ref<MutationRecord>> records;
        UncheckedKeyHashSet<GCReachableRef<Node>> pendingTargets;
    };
    TakenRecords takeRecords();
    void disconnect();

    void observationStarted(MutationObserverRegistration&);
    void observationEnded(MutationObserverRegistration&);
    void enqueueMutationRecord(Ref<MutationRecord>&&);
    void setHasTransientRegistration(Document&);
    bool canDeliver();

    bool isReachableFromOpaqueRoots(JSC::AbstractSlotVisitor&) const;

    MutationCallback& callback() const { return m_callback.get(); }
    Ref<MutationCallback> protectedCallback() const;

    static void enqueueSlotChangeEvent(HTMLSlotElement&);

    static void notifyMutationObservers(WindowEventLoop&);

    using OptionType = MutationObserverOptionType;

    static constexpr MutationObserverOptions AllMutationTypes { OptionType::ChildList, OptionType::Attributes, OptionType::CharacterData };
    static constexpr MutationObserverOptions AllDeliveryFlags { OptionType::AttributeOldValue, OptionType::CharacterDataOldValue };

private:
    explicit MutationObserver(Ref<MutationCallback>&&);
    void deliver();

    static bool validateOptions(MutationObserverOptions);

    Ref<MutationCallback> m_callback;
    Vector<Ref<MutationRecord>> m_records;
    UncheckedKeyHashSet<GCReachableRef<Node>> m_pendingTargets;
    WeakHashSet<MutationObserverRegistration> m_registrations;
    unsigned m_priority;
};

} // namespace WebCore

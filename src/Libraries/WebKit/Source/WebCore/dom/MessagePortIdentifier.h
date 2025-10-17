/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 14, 2024.
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

#include "PortIdentifier.h"
#include "ProcessIdentifier.h"
#include <wtf/Hasher.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

struct MessagePortIdentifier {
    ProcessIdentifier processIdentifier;
    PortIdentifier portIdentifier;

    friend bool operator==(const MessagePortIdentifier&, const MessagePortIdentifier&) = default;

#if !LOG_DISABLED
    String logString() const;
#endif
};

inline void add(Hasher& hasher, const MessagePortIdentifier& identifier)
{
    add(hasher, identifier.processIdentifier, identifier.portIdentifier);
}

#if !LOG_DISABLED

inline String MessagePortIdentifier::logString() const
{
    return makeString(processIdentifier.toUInt64(), '-', portIdentifier.toUInt64());
}

#endif

} // namespace WebCore

namespace WTF {

struct MessagePortIdentifierHash {
    static unsigned hash(const WebCore::MessagePortIdentifier& key) { return computeHash(key); }
    static bool equal(const WebCore::MessagePortIdentifier& a, const WebCore::MessagePortIdentifier& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = true;
};

template<> struct HashTraits<WebCore::MessagePortIdentifier> : GenericHashTraits<WebCore::MessagePortIdentifier> {
    static WebCore::MessagePortIdentifier emptyValue() { return { HashTraits<WebCore::ProcessIdentifier>::emptyValue(), HashTraits<WebCore::PortIdentifier>::emptyValue() }; }
    static bool isEmptyValue(const WebCore::MessagePortIdentifier& value) { return value.portIdentifier.isHashTableEmptyValue(); }

    static void constructDeletedValue(WebCore::MessagePortIdentifier& slot) { new (NotNull, &slot.processIdentifier) WebCore::ProcessIdentifier(WTF::HashTableDeletedValue); }

    static bool isDeletedValue(const WebCore::MessagePortIdentifier& slot) { return slot.processIdentifier.isHashTableDeletedValue(); }
};

template<> struct DefaultHash<WebCore::MessagePortIdentifier> : MessagePortIdentifierHash { };

} // namespace WTF

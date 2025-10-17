/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 9, 2023.
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

#include "ProcessIdentifier.h"
#include <wtf/HashTraits.h>
#include <wtf/Hasher.h>
#include <wtf/ObjectIdentifier.h>

namespace WebCore {

enum class WindowIdentifierType { };
using WindowIdentifier = ObjectIdentifier<WindowIdentifierType>;

// Window identifier that is unique across all WebContent processes.
struct GlobalWindowIdentifier {
    ProcessIdentifier processIdentifier;
    WindowIdentifier windowIdentifier;

    friend bool operator==(const GlobalWindowIdentifier&, const GlobalWindowIdentifier&) = default;
};

inline void add(Hasher& hasher, const GlobalWindowIdentifier& identifier)
{
    add(hasher, identifier.processIdentifier, identifier.windowIdentifier);
}

} // namespace WebCore

namespace WTF {

struct GlobalWindowIdentifierHash {
    static unsigned hash(const WebCore::GlobalWindowIdentifier& key) { return computeHash(key); }
    static bool equal(const WebCore::GlobalWindowIdentifier& a, const WebCore::GlobalWindowIdentifier& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = true;
};

template<> struct HashTraits<WebCore::GlobalWindowIdentifier> : GenericHashTraits<WebCore::GlobalWindowIdentifier> {
    static WebCore::GlobalWindowIdentifier emptyValue() { return { HashTraits<WebCore::ProcessIdentifier>::emptyValue(), HashTraits<WebCore::WindowIdentifier>::emptyValue() }; }
    static bool isEmptyValue(const WebCore::GlobalWindowIdentifier& value) { return value.windowIdentifier.isHashTableEmptyValue(); }

    static void constructDeletedValue(WebCore::GlobalWindowIdentifier& slot)
    {
        new (NotNull, &slot.processIdentifier) WebCore::ProcessIdentifier(WTF::HashTableDeletedValue);
        new (NotNull, &slot.windowIdentifier) WebCore::WindowIdentifier(WTF::HashTableDeletedValue);
    }
    static bool isDeletedValue(const WebCore::GlobalWindowIdentifier& slot) { return slot.windowIdentifier.isHashTableDeletedValue(); }
};

template<> struct DefaultHash<WebCore::GlobalWindowIdentifier> : GlobalWindowIdentifierHash { };

} // namespace WTF

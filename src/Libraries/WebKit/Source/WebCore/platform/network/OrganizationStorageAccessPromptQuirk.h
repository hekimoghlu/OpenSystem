/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 18, 2024.
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

#include "RegistrableDomain.h"
#include <wtf/CrossThreadCopier.h>
#include <wtf/HashMap.h>
#include <wtf/Vector.h>

namespace WebCore {

struct OrganizationStorageAccessPromptQuirk {
    String organizationName;
    HashMap<RegistrableDomain, Vector<RegistrableDomain>> quirkDomains;
    Vector<URL> triggerPages;

    bool isHashTableDeletedValue() const { return organizationName.isHashTableDeletedValue(); }

    OrganizationStorageAccessPromptQuirk(String&& organizationName, HashMap<RegistrableDomain, Vector<RegistrableDomain>>&& quirkDomains, Vector<URL>&& triggerPages)
        : organizationName { WTFMove(organizationName) }
        , quirkDomains { WTFMove(quirkDomains) }
        , triggerPages { WTFMove(triggerPages) }
        { }

    OrganizationStorageAccessPromptQuirk(WTF::HashTableDeletedValueType)
        : organizationName { WTF::HashTableDeletedValue }
        { }

    OrganizationStorageAccessPromptQuirk() = default;
    OrganizationStorageAccessPromptQuirk isolatedCopy() const &
    {
        return {
            crossThreadCopy(organizationName),
            crossThreadCopy(quirkDomains),
            crossThreadCopy(triggerPages)
        };
    }

    OrganizationStorageAccessPromptQuirk isolatedCopy() &&
    {
        return {
            crossThreadCopy(WTFMove(organizationName)),
            crossThreadCopy(WTFMove(quirkDomains)),
            crossThreadCopy(WTFMove(triggerPages))
        };
    }
};

static bool operator==(const OrganizationStorageAccessPromptQuirk& a, const OrganizationStorageAccessPromptQuirk& b)
{
    return a.organizationName == b.organizationName;
}

inline void add(Hasher& hasher, const OrganizationStorageAccessPromptQuirk& quirk)
{
    add(hasher, quirk.organizationName);
}

struct OrganizationStorageAccessPromptQuirkHashTraits : SimpleClassHashTraits<OrganizationStorageAccessPromptQuirk> {
    static const bool hasIsEmptyValueFunction = true;
    static const bool emptyValueIsZero = false;
    static bool isEmptyValue(const OrganizationStorageAccessPromptQuirk& quirk) { return quirk.organizationName.isNull(); }
};

struct OrganizationStorageAccessPromptQuirkHash {
    static unsigned hash(const OrganizationStorageAccessPromptQuirk& quirk) { return computeHash(quirk); }
    static bool equal(const OrganizationStorageAccessPromptQuirk& a, const OrganizationStorageAccessPromptQuirk& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = false;
};

} // namespace WebCore

namespace WTF {

template<> struct HashTraits<WebCore::OrganizationStorageAccessPromptQuirk> : WebCore::OrganizationStorageAccessPromptQuirkHashTraits { };
template<> struct DefaultHash<WebCore::OrganizationStorageAccessPromptQuirk> : WebCore::OrganizationStorageAccessPromptQuirkHash { };

} // namespace WTF

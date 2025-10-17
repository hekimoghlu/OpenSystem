/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 28, 2022.
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

#include "CacheableIdentifier.h"
#include "SlotVisitorMacros.h"
#include <wtf/Box.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

class SetPrivateBrandStatus;

class SetPrivateBrandVariant {
    WTF_MAKE_TZONE_ALLOCATED(SetPrivateBrandVariant);
public:
    SetPrivateBrandVariant(CacheableIdentifier, Structure* oldStructure, Structure* newStructure);

    ~SetPrivateBrandVariant();

    Structure* oldStructure() const { return m_oldStructure; }
    Structure* newStructure() const { return m_newStructure; }

    bool attemptToMerge(const SetPrivateBrandVariant& other);

    DECLARE_VISIT_AGGREGATE;
    template<typename Visitor> void markIfCheap(Visitor&);
    bool finalize(VM&);

    void dump(PrintStream&) const;
    void dumpInContext(PrintStream&, DumpContext*) const;

    CacheableIdentifier identifier() const { return m_identifier; }

    bool overlaps(const SetPrivateBrandVariant& other)
    {
        if (!!m_identifier != !!other.m_identifier)
            return true;
        if (m_identifier) {
            if (m_identifier != other.m_identifier)
                return false;
        }
        return m_oldStructure == other.m_oldStructure;
    }

private:
    friend class SetPrivateBrandStatus;

    Structure* m_oldStructure;
    Structure* m_newStructure;
    CacheableIdentifier m_identifier;
};

} // namespace JSC

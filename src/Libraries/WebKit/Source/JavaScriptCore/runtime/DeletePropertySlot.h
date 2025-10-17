/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 13, 2021.
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

#include "PropertyOffset.h"
#include "PropertySlot.h"

namespace JSC {
    
class DeletePropertySlot {
public:
    enum Type : uint8_t { Uncacheable, DeleteHit, ConfigurableDeleteMiss, Nonconfigurable };

    DeletePropertySlot()
        : m_offset(invalidOffset)
        , m_cacheability(CachingAllowed)
        , m_type(Uncacheable)
    {
    }

    void setConfigurableMiss()
    {
        m_type = ConfigurableDeleteMiss;
    }

    void setNonconfigurable()
    {
        m_type = Nonconfigurable;
    }

    void setHit(PropertyOffset offset)
    {
        m_type = DeleteHit;
        m_offset = offset;
    }

    bool isCacheableDelete() const { return isCacheable() && m_type != Uncacheable; }
    bool isDeleteHit() const { return m_type == DeleteHit; }
    bool isConfigurableDeleteMiss() const { return m_type == ConfigurableDeleteMiss; }
    bool isNonconfigurable() const { return m_type == Nonconfigurable; }

    PropertyOffset cachedOffset() const
    {
        return m_offset;
    }

    void disableCaching()
    {
        m_cacheability = CachingDisallowed;
    }

private:
    bool isCacheable() const { return m_cacheability == CachingAllowed; }

    PropertyOffset m_offset;
    CacheabilityType m_cacheability;
    Type m_type;
};

} // namespace JSC

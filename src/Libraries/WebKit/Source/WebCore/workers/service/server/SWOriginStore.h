/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 10, 2022.
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

#include "SecurityOriginData.h"
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SWOriginStore {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(SWOriginStore, WEBCORE_EXPORT);
public:
    virtual ~SWOriginStore() = default;

    void add(const SecurityOriginData&);
    void remove(const SecurityOriginData&);
    void clear(const SecurityOriginData&);
    void clearAll();

    virtual void importComplete() = 0;

private:
    virtual void addToStore(const SecurityOriginData&) = 0;
    virtual void removeFromStore(const SecurityOriginData&) = 0;
    virtual void clearStore() = 0;

    HashMap<SecurityOriginData, uint64_t> m_originCounts;
};

} // namespace WebCore

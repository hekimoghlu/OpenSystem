/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 5, 2023.
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

#include "DeferredTrigger.h"

namespace bmalloc {

template<IsoPageTrigger trigger>
template<typename Config>
void DeferredTrigger<trigger>::didBecome(IsoPage<Config>& page)
{
    if (page.isInUseForAllocation())
        m_hasBeenDeferred = true;
    else
        page.directory().didBecome(&page, trigger);
}

template<IsoPageTrigger trigger>
template<typename Config>
void DeferredTrigger<trigger>::handleDeferral(IsoPage<Config>& page)
{
    RELEASE_BASSERT(!page.isInUseForAllocation());
    
    if (m_hasBeenDeferred) {
        page.directory().didBecome(&page, trigger);
        m_hasBeenDeferred = false;
    }
}

} // namespace bmalloc


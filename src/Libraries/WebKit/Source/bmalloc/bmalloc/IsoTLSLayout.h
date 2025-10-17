/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 29, 2022.
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

#if !BUSE(TZONE)

#include "Mutex.h"
#include "StaticPerProcess.h"
#include <mutex>

#if !BUSE(LIBPAS)

namespace bmalloc {

class IsoTLSEntry;

class IsoTLSLayout : public StaticPerProcess<IsoTLSLayout> {
public:
    BEXPORT IsoTLSLayout(const LockHolder&);
    
    BEXPORT void add(IsoTLSEntry*);
    
    IsoTLSEntry* head() const { return m_head; }
    
private:
    IsoTLSEntry* m_head { nullptr };
    IsoTLSEntry* m_tail { nullptr };
};
BALLOW_DEPRECATED_DECLARATIONS_BEGIN
DECLARE_STATIC_PER_PROCESS_STORAGE(IsoTLSLayout);
BALLOW_DEPRECATED_DECLARATIONS_END

} // namespace bmalloc

#endif
#endif // !BUSE(TZONE)


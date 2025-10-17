/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 5, 2024.
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

#if ENABLE(JIT)

#include "CCallHelpers.h"
#include <wtf/HashMap.h>
#include <wtf/PrintStream.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace JSC {

class JITPlan;

class JITSizeStatistics {
    WTF_MAKE_TZONE_ALLOCATED(JITSizeStatistics);
public:
    struct Marker {
        String identifier;
        CCallHelpers::Label start;
    };

    Marker markStart(String identifier, CCallHelpers&);
    void markEnd(Marker, CCallHelpers&, JITPlan&);

    JS_EXPORT_PRIVATE void dump(PrintStream&) const;

    void reset() { m_data.clear(); }

private:
    struct Entry {
        size_t count { 0 };
        size_t totalBytes { 0 };
    };

    UncheckedKeyHashMap<String, Entry> m_data;
};

} // namespace JSC

#endif

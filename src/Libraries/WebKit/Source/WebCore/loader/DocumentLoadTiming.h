/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 13, 2024.
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

#include "ResourceLoadTiming.h"

namespace WebCore {

class DocumentLoadTiming : public ResourceLoadTiming {
public:
    // https://www.w3.org/TR/hr-time-2/#dfn-time-origin
    MonotonicTime timeOrigin() const { return startTime(); }

    void markUnloadEventStart() { m_unloadEventStart = MonotonicTime::now(); }
    void markUnloadEventEnd() { m_unloadEventEnd = MonotonicTime::now(); }
    void setLoadEventStart(MonotonicTime time) { m_loadEventStart = time; }
    void setLoadEventEnd(MonotonicTime time) { m_loadEventEnd = time; }

    void setHasSameOriginAsPreviousDocument(bool value) { m_hasSameOriginAsPreviousDocument = value; }

    MonotonicTime unloadEventStart() const { return m_unloadEventStart; }
    MonotonicTime unloadEventEnd() const { return m_unloadEventEnd; }
    MonotonicTime loadEventStart() const { return m_loadEventStart; }
    MonotonicTime loadEventEnd() const { return m_loadEventEnd; }
    bool hasSameOriginAsPreviousDocument() const { return m_hasSameOriginAsPreviousDocument; }

private:
    MonotonicTime m_unloadEventStart;
    MonotonicTime m_unloadEventEnd;
    MonotonicTime m_loadEventStart;
    MonotonicTime m_loadEventEnd;
    bool m_hasSameOriginAsPreviousDocument { false };
};

} // namespace WebCore

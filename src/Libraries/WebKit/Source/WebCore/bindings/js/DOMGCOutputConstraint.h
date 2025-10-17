/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 11, 2025.
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

#include <JavaScriptCore/MarkingConstraint.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {
class VM;
}

namespace WebCore {

class JSHeapData;

class DOMGCOutputConstraint : public JSC::MarkingConstraint {
    WTF_MAKE_TZONE_ALLOCATED(DOMGCOutputConstraint);
public:
    DOMGCOutputConstraint(JSC::VM&, JSHeapData&);
    ~DOMGCOutputConstraint();
    
protected:
    void executeImpl(JSC::AbstractSlotVisitor&) override;
    void executeImpl(JSC::SlotVisitor&) override;

private:
    template<typename Visitor> void executeImplImpl(Visitor&);

    JSC::VM& m_vm;
    JSHeapData& m_heapData;
    uint64_t m_lastExecutionVersion;
};

} // namespace WebCore

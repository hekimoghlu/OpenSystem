/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 21, 2023.
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

#include "RootMarkReason.h"
#include <wtf/text/UniquedStringImpl.h>
#include <wtf/text/WTFString.h>

namespace JSC {

class JSCell;

class JS_EXPORT_PRIVATE HeapAnalyzer {
public:
    virtual ~HeapAnalyzer() = default;

    // A root or marked cell.
    virtual void analyzeNode(JSCell*) = 0;

    // A reference from one cell to another.
    virtual void analyzeEdge(JSCell* from, JSCell* to, RootMarkReason) = 0;
    virtual void analyzePropertyNameEdge(JSCell* from, JSCell* to, UniquedStringImpl* propertyName) = 0;
    virtual void analyzeVariableNameEdge(JSCell* from, JSCell* to, UniquedStringImpl* variableName) = 0;
    virtual void analyzeIndexEdge(JSCell* from, JSCell* to, uint32_t index) = 0;

    virtual void setOpaqueRootReachabilityReasonForCell(JSCell*, ASCIILiteral) = 0;
    virtual void setWrappedObjectForCell(JSCell*, void*) = 0;
    virtual void setLabelForCell(JSCell*, const String&) = 0;
};

} // namespace JSC

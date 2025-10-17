/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 12, 2025.
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

#if ENABLE(B3_JIT)

#include "B3Value.h"
#include <wtf/IndexSet.h>
#include <wtf/Vector.h>

namespace JSC { namespace B3 {

class Procedure;

// Turns all mentions of the given values into accesses to variables. This is meant to be used
// from phases that don't like SSA for whatever reason.
JS_EXPORT_PRIVATE void demoteValues(Procedure&, const IndexSet<Value*>&);

// This fixes SSA for you. Use this after you have done demoteValues() and you have performed
// whatever evil transformation you needed.
JS_EXPORT_PRIVATE bool fixSSA(Procedure&);

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

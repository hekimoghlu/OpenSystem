/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 23, 2021.
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

#include <wtf/Forward.h>

namespace JSC {

#define FOR_EACH_ROOT_MARK_REASON(v) \
    v(None) \
    v(ConservativeScan) \
    v(ExecutableToCodeBlockEdges) \
    v(ExternalRememberedSet) \
    v(StrongReferences) \
    v(ProtectedValues) \
    v(MarkedJSValueRefArray) \
    v(MarkListSet) \
    v(VMExceptions) \
    v(StrongHandles) \
    v(Debugger) \
    v(JITStubRoutines) \
    v(WeakMapSpace) \
    v(WeakSets) \
    v(Output) \
    v(JITWorkList) \
    v(CodeBlocks) \
    v(DOMGCOutput)

#define DECLARE_ROOT_MARK_REASON(reason) reason,

enum class RootMarkReason : uint8_t {
    FOR_EACH_ROOT_MARK_REASON(DECLARE_ROOT_MARK_REASON)
};

#undef DECLARE_ROOT_MARK_REASON

ASCIILiteral rootMarkReasonDescription(RootMarkReason);

} // namespace JSC

namespace WTF {

class PrintStream;

void printInternal(PrintStream&, JSC::RootMarkReason);

} // namespace WTF

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 15, 2024.
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
#include "config.h"
#include "DFGDoesGCCheck.h"

#include "CallFrameInlines.h"
#include "CodeBlock.h"
#include "DFGGraph.h"
#include "DFGNodeType.h"
#include "JSCJSValueInlines.h"
#include "Options.h"
#include "VMInspector.h"
#include <wtf/DataLog.h>

namespace JSC {
namespace DFG {

#if ENABLE(DFG_DOES_GC_VALIDATION)

void DoesGCCheck::verifyCanGC(VM& vm)
{
    // We do this check here just so we don't have to #include DFGNodeType.h
    // in the header file.
    static_assert(numberOfNodeTypes <= (1 << nodeOpBits));

    if (!Options::validateDoesGC())
        return;

    if (!expectDoesGC()) {
        dataLog("Error: DoesGC failed");
        if (isSpecial()) {
            switch (special()) {
            case Special::Uninitialized:
                break;
            case Special::DFGOSRExit:
                dataLog(" @ DFG osr exit");
                break;
            case Special::FTLOSRExit:
                dataLog(" @ FTL osr exit");
                break;
            case Special::NumberOfSpecials:
                RELEASE_ASSERT_NOT_REACHED();
            }
        } else
            dataLog(" @ D@", nodeIndex(), " ", DFG::Graph::opName(static_cast<DFG::NodeType>(nodeOp())));

        CallFrame* callFrame = vm.topCallFrame;
        if (callFrame) {
            if (!callFrame->isNativeCalleeFrame())
                dataLogLn(" in ", callFrame->codeBlock());
            VMInspector::dumpStack(&vm, callFrame);
        }
        dataLogLn();
    }
    RELEASE_ASSERT(expectDoesGC());
}
#endif // ENABLE(DFG_DOES_GC_VALIDATION)

} // namespace DFG
} // namespace JSC


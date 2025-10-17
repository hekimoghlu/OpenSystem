/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 17, 2022.
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
#include "DFGDesiredGlobalProperties.h"

#if ENABLE(DFG_JIT)

#include "CodeBlock.h"
#include "DFGCommonData.h"
#include "DFGDesiredIdentifiers.h"
#include "DFGDesiredWatchpoints.h"
#include "JSGlobalObject.h"

namespace JSC { namespace DFG {

bool DesiredGlobalProperties::isStillValidOnMainThread(VM& vm, DesiredIdentifiers& identifiers)
{
    bool isStillValid = true;
    for (const auto& property : m_set) {
        auto* uid = identifiers.at(property.identifierNumber());
        JSGlobalObject* globalObject = property.globalObject();
        {
            SymbolTable* symbolTable = globalObject->globalLexicalEnvironment()->symbolTable();
            ConcurrentJSLocker locker(symbolTable->m_lock);
            if (!symbolTable->contains(locker, uid))
                continue;
        }
        // Set invalidated WatchpointSet here to prevent further compile-and-fail loop.
        property.globalObject()->ensureReferencedPropertyWatchpointSet(uid).fireAll(vm, "Lexical binding shadows an existing global property");
        isStillValid = false;
    }
    return isStillValid;
}

bool DesiredGlobalProperties::reallyAdd(CodeBlock* codeBlock, DesiredIdentifiers& identifiers, WatchpointCollector& collector)
{
    for (const auto& property : m_set) {
        bool result = collector.addWatchpoint([&](CodeBlockJettisoningWatchpoint& watchpoint) {
            auto* uid = identifiers.at(property.identifierNumber());
            JSGlobalObject* globalObject = property.globalObject();
            {
                ConcurrentJSLocker locker(codeBlock->m_lock);
                watchpoint.initialize(codeBlock);
            }
            auto& watchpointSet = globalObject->ensureReferencedPropertyWatchpointSet(uid);
            ASSERT(watchpointSet.isStillValid());
            watchpointSet.add(&watchpoint);
            return true;
        });
        if (!result)
            return false;
    }
    return true;
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)


/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 25, 2024.
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
#include "WasmHandlerInfo.h"

#if ENABLE(WEBASSEMBLY)

#include "JSWebAssemblyInstance.h"
#include "WasmTag.h"

namespace JSC {
namespace Wasm {

void HandlerInfo::initialize(const UnlinkedHandlerInfo& unlinkedInfo, CodePtr<ExceptionHandlerPtrTag> label)
{
    m_type = unlinkedInfo.m_type;
    m_start = unlinkedInfo.m_start;
    m_end = unlinkedInfo.m_end;
    m_target = unlinkedInfo.m_target;
    m_targetMetadata = unlinkedInfo.m_targetMetadata;
    m_tryDepth = unlinkedInfo.m_tryDepth;
    m_nativeCode = label;

    switch (m_type) {
    case HandlerType::Catch:
    case HandlerType::TryTableCatch:
    case HandlerType::TryTableCatchRef:
        m_tag = unlinkedInfo.m_exceptionIndexOrDelegateTarget;
        break;

    case HandlerType::CatchAll:
    case HandlerType::TryTableCatchAll:
    case HandlerType::TryTableCatchAllRef:
        break;

    case HandlerType::Delegate:
        m_delegateTarget = unlinkedInfo.m_exceptionIndexOrDelegateTarget;
        break;
    }
}

const HandlerInfo* HandlerInfo::handlerForIndex(JSWebAssemblyInstance& instance, const FixedVector<HandlerInfo>& exceptionHandlers, unsigned index, const Wasm::Tag* exceptionTag)
{
    bool delegating = false;
    unsigned delegateTarget = 0;
    for (auto& handler : exceptionHandlers) {
        // Handlers are ordered innermost first, so the first handler we encounter
        // that contains the source address is the correct handler to use.
        // This index used is either the BytecodeOffset or a CallSiteIndex.
        if (handler.m_start <= index && handler.m_end > index) {
            if (delegating) {
                if (handler.m_tryDepth != delegateTarget)
                    continue;
                delegating = false;
            }

            bool match = false;
            switch (handler.m_type) {
            case HandlerType::Catch:
            case HandlerType::TryTableCatch:
            case HandlerType::TryTableCatchRef:
                match = exceptionTag && instance.tag(handler.m_tag) == *exceptionTag;
                break;
            case HandlerType::CatchAll:
            case HandlerType::TryTableCatchAll:
            case HandlerType::TryTableCatchAllRef:
                match = true;
                break;
            case HandlerType::Delegate:
                delegating = true;
                delegateTarget = handler.m_delegateTarget;
                break;
            }

            if (!match)
                continue;

            return &handler;
        }
    }

    return nullptr;
}

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)

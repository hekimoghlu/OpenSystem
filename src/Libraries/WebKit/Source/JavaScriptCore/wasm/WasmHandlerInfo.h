/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 12, 2025.
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

#if ENABLE(WEBASSEMBLY)

#include "CodeLocation.h"
#include <wtf/Forward.h>
#include <wtf/text/ASCIILiteral.h>

namespace JSC {
class JSWebAssemblyInstance;

namespace Wasm {

class Tag;

enum class HandlerType {
    Catch = 0,
    CatchAll = 1,
    Delegate = 2,
    TryTableCatch = 3,
    TryTableCatchRef = 4,
    TryTableCatchAll = 5,
    TryTableCatchAllRef = 6,
};

struct HandlerInfoBase {
    HandlerType m_type;
    uint32_t m_start;
    uint32_t m_end;
    uint32_t m_target;
    uint32_t m_targetMetadata { 0 };
    uint32_t m_tryDepth;
    uint32_t m_exceptionIndexOrDelegateTarget;
};

struct UnlinkedHandlerInfo : public HandlerInfoBase {
    UnlinkedHandlerInfo()
    {
    }

    UnlinkedHandlerInfo(HandlerType handlerType, uint32_t start, uint32_t end, uint32_t target, uint32_t tryDepth, uint32_t exceptionIndexOrDelegateTarget)
    {
        m_type = handlerType;
        m_start = start;
        m_end = end;
        m_target = target;
        m_tryDepth = tryDepth;
        m_exceptionIndexOrDelegateTarget = exceptionIndexOrDelegateTarget;
    }

    UnlinkedHandlerInfo(HandlerType handlerType, uint32_t start, uint32_t end, uint32_t target, uint32_t targetMetadata, uint32_t tryDepth, uint32_t exceptionIndexOrDelegateTarget)
    {
        m_type = handlerType;
        m_start = start;
        m_end = end;
        m_target = target;
        m_targetMetadata = targetMetadata;
        m_tryDepth = tryDepth;
        m_exceptionIndexOrDelegateTarget = exceptionIndexOrDelegateTarget;
    }

    ASCIILiteral typeName() const
    {
        switch (m_type) {
        case HandlerType::Catch:
            return "catch"_s;
        case HandlerType::CatchAll:
            return "catchall"_s;
        case HandlerType::Delegate:
            return "delegate"_s;
        case HandlerType::TryTableCatch:
            return "try_table catch"_s;
        case HandlerType::TryTableCatchRef:
            return "try_table catch_ref"_s;
        case HandlerType::TryTableCatchAll:
            return "try_table catch_all"_s;
        case HandlerType::TryTableCatchAllRef:
            return "try_table catch_all_ref"_s;
        default:
            ASSERT_NOT_REACHED();
            break;
        }
        return { };
    }
};

struct HandlerInfo : public HandlerInfoBase {
    static const HandlerInfo* handlerForIndex(JSWebAssemblyInstance&, const FixedVector<HandlerInfo>& exeptionHandlers, unsigned index, const Wasm::Tag* exceptionTag);

    void initialize(const UnlinkedHandlerInfo&, CodePtr<ExceptionHandlerPtrTag>);

    unsigned tag() const
    {
        ASSERT(m_type == HandlerType::Catch);
        return m_tag;
    }

    unsigned delegateTarget() const
    {
        ASSERT(m_type == HandlerType::Delegate);
        return m_delegateTarget;
    }

    CodePtr<ExceptionHandlerPtrTag> m_nativeCode;

private:
    union {
        unsigned m_tag;
        unsigned m_delegateTarget;
    };
};

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)

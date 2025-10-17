/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 24, 2021.
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

#include "CodeLocation.h"
#include <wtf/Forward.h>

namespace JSC {

enum class HandlerType : uint8_t {
    Catch = 0,
    Finally = 1,
    SynthesizedCatch = 2,
    SynthesizedFinally = 3
};

enum class RequiredHandler {
    CatchHandler,
    AnyHandler
};

struct HandlerInfoBase {
    HandlerType type() const { return static_cast<HandlerType>(typeBits); }
    void setType(HandlerType type) { typeBits = static_cast<uint32_t>(type); }

    const char* typeName() const
    {
        switch (type()) {
        case HandlerType::Catch:
            return "catch";
        case HandlerType::Finally:
            return "finally";
        case HandlerType::SynthesizedCatch:
            return "synthesized catch";
        case HandlerType::SynthesizedFinally:
            return "synthesized finally";
        default:
            ASSERT_NOT_REACHED();
        }
        return nullptr;
    }

    bool isCatchHandler() const { return type() == HandlerType::Catch; }

    template<typename Handler, typename Container>
    static Handler* handlerForIndex(Container& exeptionHandlers, unsigned index, RequiredHandler requiredHandler)
    {
        for (auto& handler : exeptionHandlers) {
            if ((requiredHandler == RequiredHandler::CatchHandler) && !handler.isCatchHandler())
                continue;

            // Handlers are ordered innermost first, so the first handler we encounter
            // that contains the source address is the correct handler to use.
            // This index used is either the BytecodeOffset or a CallSiteIndex.
            if (handler.start <= index && handler.end > index)
                return &handler;
        }

        return nullptr;
    }

    uint32_t start;
    uint32_t end;
    uint32_t target;
    uint32_t typeBits : 2; // HandlerType
};

struct UnlinkedHandlerInfo : public HandlerInfoBase {
    UnlinkedHandlerInfo()
    {
    }

    UnlinkedHandlerInfo(uint32_t start, uint32_t end, uint32_t target, HandlerType handlerType)
    {
        this->start = start;
        this->end = end;
        this->target = target;
        setType(handlerType);
        ASSERT(type() == handlerType);
    }
};

struct HandlerInfo : public HandlerInfoBase {
    void initialize(const UnlinkedHandlerInfo& unlinkedInfo)
    {
        start = unlinkedInfo.start;
        end = unlinkedInfo.end;
        target = unlinkedInfo.target;
        typeBits = unlinkedInfo.typeBits;
    }

#if ENABLE(JIT)
    void initialize(const UnlinkedHandlerInfo& unlinkedInfo, CodeLocationLabel<ExceptionHandlerPtrTag> label)
    {
        initialize(unlinkedInfo);
        nativeCode = label;
    }

    CodeLocationLabel<ExceptionHandlerPtrTag> nativeCode;
#endif
};

} // namespace JSC

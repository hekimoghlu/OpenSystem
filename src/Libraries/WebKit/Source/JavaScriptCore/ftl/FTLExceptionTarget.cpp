/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 25, 2023.
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
#include "FTLExceptionTarget.h"

#if ENABLE(FTL_JIT)

#include "LinkBuffer.h"

namespace JSC { namespace FTL {

ExceptionTarget::~ExceptionTarget() = default;

CodeLocationLabel<ExceptionHandlerPtrTag> ExceptionTarget::label(LinkBuffer& linkBuffer)
{
    if (m_isDefaultHandler)
        return linkBuffer.locationOf<ExceptionHandlerPtrTag>(*m_defaultHandler);
    return linkBuffer.locationOf<ExceptionHandlerPtrTag>(m_handle->label);
}

Box<CCallHelpers::JumpList> ExceptionTarget::jumps(CCallHelpers& jit)
{
    Box<CCallHelpers::JumpList> result = Box<CCallHelpers::JumpList>::create();
    if (m_isDefaultHandler) {
        Box<CCallHelpers::Label> defaultHandler = m_defaultHandler;
        jit.addLinkTask(
            [=] (LinkBuffer& linkBuffer) {
                linkBuffer.link(*result, linkBuffer.locationOf<ExceptionHandlerPtrTag>(*defaultHandler));
            });
    } else {
        RefPtr<OSRExitHandle> handle = m_handle;
        jit.addLinkTask(
            [=] (LinkBuffer& linkBuffer) {
                linkBuffer.link(*result, linkBuffer.locationOf<OSRExitPtrTag>(handle->label));
            });
    }
    return result;
}

ExceptionTarget::ExceptionTarget(
    bool isDefaultHandler, Box<CCallHelpers::Label> defaultHandler, RefPtr<OSRExitHandle> handle)
    : m_isDefaultHandler(isDefaultHandler)
    , m_defaultHandler(defaultHandler)
    , m_handle(handle)
{
}

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)


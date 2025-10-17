/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 16, 2024.
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

#if ENABLE(WEBGL)
#include "EXTDisjointTimerQueryWebGL2.h"

#include "EventLoop.h"
#include "ScriptExecutionContext.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(EXTDisjointTimerQueryWebGL2);

EXTDisjointTimerQueryWebGL2::EXTDisjointTimerQueryWebGL2(WebGLRenderingContextBase& context)
    : WebGLExtension(context, WebGLExtensionName::EXTDisjointTimerQueryWebGL2)
{
    context.protectedGraphicsContextGL()->ensureExtensionEnabled("GL_EXT_disjoint_timer_query"_s);
}

EXTDisjointTimerQueryWebGL2::~EXTDisjointTimerQueryWebGL2() = default;

bool EXTDisjointTimerQueryWebGL2::supported(GraphicsContextGL& context)
{
    return context.supportsExtension("GL_EXT_disjoint_timer_query"_s);
}

void EXTDisjointTimerQueryWebGL2::queryCounterEXT(WebGLQuery& query, GCGLenum target)
{
    if (isContextLost())
        return;
    auto& context = this->context();
    if (!context.scriptExecutionContext())
        return;

    if (!context.validateWebGLObject("queryCounterEXT"_s, query))
        return;

    if (target != GraphicsContextGL::TIMESTAMP_EXT) {
        context.synthesizeGLError(GraphicsContextGL::INVALID_ENUM, "queryCounterEXT"_s, "invalid target"_s);
        return;
    }

    if (query.target() && query.target() != target) {
        context.synthesizeGLError(GraphicsContextGL::INVALID_OPERATION, "queryCounterEXT"_s, "query type does not match target"_s);
        return;
    }

    query.setTarget(target);

    context.protectedGraphicsContextGL()->queryCounterEXT(query.object(), target);

    // A query's result must not be made available until control has returned to the user agent's main loop.
    context.scriptExecutionContext()->eventLoop().queueMicrotask([&] {
        query.makeResultAvailable();
    });
}

} // namespace WebCore

#endif // ENABLE(WEBGL)

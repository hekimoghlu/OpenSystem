/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 12, 2023.
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

#include "EXTDisjointTimerQuery.h"

#include "EventLoop.h"
#include "ScriptExecutionContext.h"
#include "WebGLRenderingContext.h"
#include "WebGLTimerQueryEXT.h"
#include <wtf/Lock.h>
#include <wtf/Locker.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(EXTDisjointTimerQuery);

EXTDisjointTimerQuery::EXTDisjointTimerQuery(WebGLRenderingContext& context)
    : WebGLExtension(context, WebGLExtensionName::EXTDisjointTimerQuery)
{
    context.protectedGraphicsContextGL()->ensureExtensionEnabled("GL_EXT_disjoint_timer_query"_s);
}

EXTDisjointTimerQuery::~EXTDisjointTimerQuery() = default;

bool EXTDisjointTimerQuery::supported(GraphicsContextGL& context)
{
    return context.supportsExtension("GL_EXT_disjoint_timer_query"_s);
}

RefPtr<WebGLTimerQueryEXT> EXTDisjointTimerQuery::createQueryEXT()
{
    if (isContextLost())
        return nullptr;
    auto& context = this->context();
    return WebGLTimerQueryEXT::create(context);
}

void EXTDisjointTimerQuery::deleteQueryEXT(WebGLTimerQueryEXT* query)
{
    if (isContextLost())
        return;
    auto& context = this->context();

    Locker locker { context.objectGraphLock() };

    if (!query)
        return;

    if (!query->validate(context)) {
        context.synthesizeGLError(GraphicsContextGL::INVALID_OPERATION, "delete"_s, "object does not belong to this context"_s);
        return;
    }

    if (query->isDeleted())
        return;

    if (query == context.m_activeQuery) {
        context.m_activeQuery = nullptr;
        ASSERT(query->target() == GraphicsContextGL::TIME_ELAPSED_EXT);
        context.protectedGraphicsContextGL()->endQueryEXT(GraphicsContextGL::TIME_ELAPSED_EXT);
    }

    query->deleteObject(locker, context.protectedGraphicsContextGL().get());
}

GCGLboolean EXTDisjointTimerQuery::isQueryEXT(WebGLTimerQueryEXT* query)
{
    if (isContextLost())
        return false;
    auto& context = this->context();
    if (!context.validateIsWebGLObject(query))
        return false;
    return context.protectedGraphicsContextGL()->isQueryEXT(query->object());
}

void EXTDisjointTimerQuery::beginQueryEXT(GCGLenum target, WebGLTimerQueryEXT& query)
{
    if (isContextLost())
        return;
    auto& context = this->context();

    Locker locker { context.objectGraphLock() };

    if (!context.validateWebGLObject("beginQueryEXT"_s, query))
        return;

    // The WebGL extension requires ending time elapsed queries when they are deleted.
    // Ending non-active queries is invalid so m_activeQuery is used to track them and
    // to defer query results until control is returned to the user agent's main loop.

    if (target != GraphicsContextGL::TIME_ELAPSED_EXT) {
        context.synthesizeGLError(GraphicsContextGL::INVALID_ENUM, "beginQueryEXT"_s, "invalid target"_s);
        return;
    }

    if (query.target() && query.target() != target) {
        context.synthesizeGLError(GraphicsContextGL::INVALID_OPERATION, "beginQueryEXT"_s, "query type does not match target"_s);
        return;
    }

    if (context.m_activeQuery) {
        context.synthesizeGLError(GraphicsContextGL::INVALID_OPERATION, "beginQueryEXT"_s, "query object of target is already active"_s);
        return;
    }

    context.m_activeQuery = &query;

    context.protectedGraphicsContextGL()->beginQueryEXT(target, query.object());
}

void EXTDisjointTimerQuery::endQueryEXT(GCGLenum target)
{
    if (isContextLost())
        return;
    auto& context = this->context();
    if (!context.scriptExecutionContext())
        return;

    Locker locker { context.objectGraphLock() };

    if (target != GraphicsContextGL::TIME_ELAPSED_EXT) {
        context.synthesizeGLError(GraphicsContextGL::INVALID_ENUM, "endQueryEXT"_s, "invalid target"_s);
        return;
    }

    if (!context.m_activeQuery) {
        context.synthesizeGLError(GraphicsContextGL::INVALID_OPERATION, "endQueryEXT"_s, "query object of target is not active"_s);
        return;
    }

    context.protectedGraphicsContextGL()->endQueryEXT(target);

    // A query's result must not be made available until control has returned to the user agent's main loop.
    context.scriptExecutionContext()->eventLoop().queueMicrotask([query = WTFMove(context.m_activeQuery)] {
        query->makeResultAvailable();
    });
}

void EXTDisjointTimerQuery::queryCounterEXT(WebGLTimerQueryEXT& query, GCGLenum target)
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

WebGLAny EXTDisjointTimerQuery::getQueryEXT(GCGLenum target, GCGLenum pname)
{
    if (isContextLost())
        return nullptr;
    auto& context = this->context();

    if (target != GraphicsContextGL::TIME_ELAPSED_EXT && target != GraphicsContextGL::TIMESTAMP_EXT) {
        context.synthesizeGLError(GraphicsContextGL::INVALID_ENUM, "getQueryEXT"_s, "invalid target"_s);
        return nullptr;
    }

    switch (pname) {
    case GraphicsContextGL::CURRENT_QUERY_EXT:
        if (target == GraphicsContextGL::TIME_ELAPSED_EXT)
            return context.m_activeQuery;
        return nullptr;
    case GraphicsContextGL::QUERY_COUNTER_BITS_EXT:
        return context.protectedGraphicsContextGL()->getQueryiEXT(target, pname);
    }
    context.synthesizeGLError(GraphicsContextGL::INVALID_ENUM, "getQueryEXT"_s, "invalid parameter name"_s);
    return nullptr;
}

WebGLAny EXTDisjointTimerQuery::getQueryObjectEXT(WebGLTimerQueryEXT& query, GCGLenum pname)
{
    if (isContextLost())
        return nullptr;
    auto& context = this->context();

    if (!context.validateWebGLObject("getQueryObjectEXT"_s, query))
        return nullptr;

    if (!query.target()) {
        context.synthesizeGLError(GraphicsContextGL::INVALID_OPERATION, "getQueryObjectEXT"_s, "query has not been used"_s);
        return nullptr;
    }

    if (&query == context.m_activeQuery) {
        context.synthesizeGLError(GraphicsContextGL::INVALID_OPERATION, "getQueryObjectEXT"_s, "query is currently active"_s);
        return nullptr;
    }

    switch (pname) {
    case GraphicsContextGL::QUERY_RESULT_EXT:
        if (!query.isResultAvailable())
            return 0;
        return static_cast<unsigned long long>(context.protectedGraphicsContextGL()->getQueryObjectui64EXT(query.object(), pname));
    case GraphicsContextGL::QUERY_RESULT_AVAILABLE_EXT:
        if (!query.isResultAvailable())
            return false;
        return static_cast<bool>(context.protectedGraphicsContextGL()->getQueryObjectiEXT(query.object(), pname));
    }
    context.synthesizeGLError(GraphicsContextGL::INVALID_ENUM, "getQueryObjectEXT"_s, "invalid parameter name"_s);
    return nullptr;
}

} // namespace WebCore

#endif // ENABLE(WEBGL)

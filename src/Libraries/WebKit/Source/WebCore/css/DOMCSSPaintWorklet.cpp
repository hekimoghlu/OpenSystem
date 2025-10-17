/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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
#include "DOMCSSPaintWorklet.h"

#include "DOMCSSNamespace.h"
#include "DocumentInlines.h"
#include "JSDOMPromiseDeferred.h"
#include "PaintWorkletGlobalScope.h"
#include "WorkletGlobalScopeProxy.h"
#include <wtf/text/WTFString.h>

namespace WebCore {
DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(DOMCSSPaintWorklet);

PaintWorklet& DOMCSSPaintWorklet::ensurePaintWorklet(Document& document)
{
    return document.ensurePaintWorklet();
}

DOMCSSPaintWorklet* DOMCSSPaintWorklet::from(DOMCSSNamespace& css)
{
    auto* supplement = static_cast<DOMCSSPaintWorklet*>(Supplement<DOMCSSNamespace>::from(&css, supplementName()));
    if (!supplement) {
        auto newSupplement = makeUnique<DOMCSSPaintWorklet>(css);
        supplement = newSupplement.get();
        provideTo(&css, supplementName(), WTFMove(newSupplement));
    }
    return supplement;
}

ASCIILiteral DOMCSSPaintWorklet::supplementName()
{
    return "DOMCSSPaintWorklet"_s;
}

// FIXME: Get rid of this override and rely on the standard-compliant Worklet::addModule() instead.
void PaintWorklet::addModule(const String& moduleURL, WorkletOptions&&, DOMPromiseDeferred<void>&& promise)
{
    RefPtr document = this->document();
    if (!document) {
        promise.reject(Exception { ExceptionCode::InvalidStateError, "This frame is detached"_s });
        return;
    }

    if (!document->hasBrowsingContext()) {
        promise.reject(Exception { ExceptionCode::InvalidStateError, "This document does not have a browsing context"_s });
        return;
    }

    // FIXME: We should download the source from the URL
    // https://bugs.webkit.org/show_bug.cgi?id=191136
    // PaintWorklets don't have access to any sensitive APIs so we don't bother tracking taintedness there.
    auto maybeContext = PaintWorkletGlobalScope::tryCreate(*document, ScriptSourceCode(moduleURL, JSC::SourceTaintedOrigin::Untainted));
    if (UNLIKELY(!maybeContext)) {
        promise.reject(Exception { ExceptionCode::OutOfMemoryError });
        return;
    }
    auto context = maybeContext.releaseNonNull();
    context->evaluate();

    Locker locker { context->paintDefinitionLock() };
    for (auto& name : context->paintDefinitionMap().keys())
        document->setPaintWorkletGlobalScopeForName(name, context.copyRef());
    promise.resolve();
}

Vector<Ref<WorkletGlobalScopeProxy>> PaintWorklet::createGlobalScopes()
{
    // FIXME: Add implementation.
    return { };
}

} // namespace WebCore

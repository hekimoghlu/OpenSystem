/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 13, 2022.
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

#include "CGContextStateSaver.h"
#include "GraphicsContextCG.h"
#include "GraphicsContextStateSaver.h"
#include <wtf/Noncopyable.h>

#if PLATFORM(COCOA)

#if USE(APPKIT)
OBJC_CLASS NSGraphicsContext;
#endif

namespace WebCore {

// Scoped setter for the current NSGraphicsContext for functions which call out into AppKit and rely on the
// currentContext being set.
class LocalCurrentContextSaver {
    WTF_MAKE_NONCOPYABLE(LocalCurrentContextSaver);
public:
    WEBCORE_EXPORT LocalCurrentContextSaver(CGContextRef, bool isFlipped = true);
    WEBCORE_EXPORT ~LocalCurrentContextSaver();

private:
#if USE(APPKIT)
    RetainPtr<NSGraphicsContext> m_savedNSGraphicsContext;
#endif
    bool m_didSetGraphicsContext { false };
};

// Scoped setter for the current NSGraphicsContext for functions which call out into AppKit and rely on the
// currentContext being set.
// Preserves the CGContext state.
class LocalCurrentCGContext {
    WTF_MAKE_NONCOPYABLE(LocalCurrentCGContext);
public:
    LocalCurrentCGContext(CGContextRef context)
        : m_stateSaver(context)
        , m_globalSaver(context)
    {
    }

    ~LocalCurrentCGContext() = default;

private:
    CGContextStateSaver m_stateSaver;
    LocalCurrentContextSaver m_globalSaver;
};

// Scoped setter for the current NSGraphicsContext for functions which call out into AppKit and rely on the
// currentContext being set.
// Preserves the GraphicsContext state.
class LocalCurrentGraphicsContext {
    WTF_MAKE_NONCOPYABLE(LocalCurrentGraphicsContext);
public:
    LocalCurrentGraphicsContext(GraphicsContext& context, bool isFlipped = true)
        : m_stateSaver(context)
        , m_globalSaver(context.platformContext(), isFlipped)
    {
    }

    ~LocalCurrentGraphicsContext() = default;

    CGContextRef cgContext() { return m_stateSaver.context()->platformContext(); }

private:
    GraphicsContextStateSaver m_stateSaver;
    LocalCurrentContextSaver m_globalSaver;
};

}

#endif // PLATFORM(COCOA)

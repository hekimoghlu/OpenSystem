/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 22, 2021.
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

#include "DestinationColorSpace.h"
#include "FloatRect.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class Filter;
class FilterResults;
class GraphicsContext;

class GraphicsContextSwitcher {
    WTF_MAKE_TZONE_ALLOCATED(GraphicsContextSwitcher);
public:
    static std::unique_ptr<GraphicsContextSwitcher> create(GraphicsContext& destinationContext, const FloatRect& sourceImageRect, const DestinationColorSpace&, RefPtr<Filter>&& = nullptr, FilterResults* = nullptr);

    virtual ~GraphicsContextSwitcher() = default;

    virtual GraphicsContext* drawingContext(GraphicsContext& destinationContext) const { return &destinationContext; }

    virtual bool hasSourceImage() const { return false; }

    virtual void beginClipAndDrawSourceImage(GraphicsContext& destinationContext, const FloatRect& repaintRect, const FloatRect& clipRect) = 0;
    virtual void endClipAndDrawSourceImage(GraphicsContext& destinationContext, const DestinationColorSpace&) = 0;

    virtual void beginDrawSourceImage(GraphicsContext& destinationContext, float opacity = 1.f) = 0;
    virtual void endDrawSourceImage(GraphicsContext& destinationContext, const DestinationColorSpace&) = 0;

protected:
    explicit GraphicsContextSwitcher(RefPtr<Filter>&&);

    RefPtr<Filter> m_filter;
};

} // namespace WebCore

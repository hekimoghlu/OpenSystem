/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 19, 2023.
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

#include "FloatSize.h"
#include <wtf/Forward.h>

namespace WebCore {

class Document;

enum class ViewportErrorCode : uint8_t {
    UnrecognizedViewportArgumentKey,
    UnrecognizedViewportArgumentValue,
    TruncatedViewportArgumentValue,
    MaximumScaleTooLarge,
};

enum class ViewportFit : uint8_t {
    Auto,
    Contain,
    Cover
};

struct ViewportAttributes {
    FloatSize layoutSize;

    float initialScale;
    float minimumScale;
    float maximumScale;

    float userScalable;
    float orientation;
    float shrinkToFit;

    ViewportFit viewportFit;
};

struct ViewportArguments {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    enum class Type : uint8_t {
        // These are ordered in increasing importance.
        Implicit,
#if PLATFORM(IOS_FAMILY)
        PluginDocument,
        ImageDocument,
#endif
        ViewportMeta,
    } type;

    static constexpr int ValueAuto = -1;
    static constexpr int ValueDeviceWidth = -2;
    static constexpr int ValueDeviceHeight = -3;
    static constexpr int ValuePortrait = -4;
    static constexpr int ValueLandscape = -5;

    explicit ViewportArguments(Type type = Type::Implicit)
        : type(type)
    {
    }

    ViewportArguments(ViewportArguments&&) = default;
    ViewportArguments(const ViewportArguments&) = default;
    ViewportArguments& operator=(ViewportArguments&&) = default;
    ViewportArguments& operator=(const ViewportArguments&) = default;

    ViewportArguments(Type type, float width, float height, float zoom, float minZoom, float maxZoom, float userZoom, float orientation, float shrinkToFit, ViewportFit viewportFit, bool widthWasExplicit)
        : type(type)
        , width(width)
        , height(height)
        , zoom(zoom)
        , minZoom(minZoom)
        , maxZoom(maxZoom)
        , userZoom(userZoom)
        , orientation(orientation)
        , shrinkToFit(shrinkToFit)
        , viewportFit(viewportFit)
        , widthWasExplicit(widthWasExplicit)
    {
    }

    // All arguments are in CSS units.
    ViewportAttributes resolve(const FloatSize& initialViewportSize, const FloatSize& deviceSize, int defaultWidth) const;

    float width { ValueAuto };
    float height { ValueAuto };
    float zoom { ValueAuto };
    float minZoom { ValueAuto };
    float maxZoom { ValueAuto };
    float userZoom { ValueAuto };
    float orientation { ValueAuto };
    float shrinkToFit { ValueAuto };
    ViewportFit viewportFit { ViewportFit::Auto };
    bool widthWasExplicit { false };

    bool operator==(const ViewportArguments& other) const
    {
        // Used for figuring out whether to reset the viewport or not,
        // thus we are not taking type into account.
        return width == other.width
            && height == other.height
            && zoom == other.zoom
            && minZoom == other.minZoom
            && maxZoom == other.maxZoom
            && userZoom == other.userZoom
            && orientation == other.orientation
            && shrinkToFit == other.shrinkToFit
            && viewportFit == other.viewportFit
            && widthWasExplicit == other.widthWasExplicit;
    }

#if PLATFORM(GTK)
    // FIXME: We're going to keep this constant around until all embedders
    // refactor their code to no longer need it.
    static const float deprecatedTargetDPI;
#endif
};

WEBCORE_EXPORT ViewportAttributes computeViewportAttributes(ViewportArguments args, int desktopWidth, int deviceWidth, int deviceHeight, float devicePixelRatio, IntSize visibleViewport);

WEBCORE_EXPORT void restrictMinimumScaleFactorToViewportSize(ViewportAttributes& result, IntSize visibleViewport, float devicePixelRatio);
WEBCORE_EXPORT void restrictScaleFactorToInitialScaleIfNotUserScalable(ViewportAttributes& result);
WEBCORE_EXPORT float computeMinimumScaleFactorForContentContained(const ViewportAttributes& result, const IntSize& viewportSize, const IntSize& contentSize);

typedef Function<void(ViewportErrorCode, const String&)> ViewportErrorHandler;
void setViewportFeature(ViewportArguments&, Document&, StringView key, StringView value);
WEBCORE_EXPORT void setViewportFeature(ViewportArguments&, StringView key, StringView value, const ViewportErrorHandler&);

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const ViewportArguments&);

} // namespace WebCore

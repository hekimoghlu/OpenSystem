/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 24, 2025.
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

//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Overlay.h:
//    Defines the Overlay class that handles overlay widgets.
//

#ifndef LIBANGLE_OVERLAY_H_
#define LIBANGLE_OVERLAY_H_

#include "common/PackedEnums.h"
#include "common/angleutils.h"
#include "libANGLE/Error.h"
#include "libANGLE/OverlayWidgets.h"
#include "libANGLE/angletypes.h"

namespace rx
{
class OverlayImpl;
class GLImplFactory;
}  // namespace rx

namespace gl
{
class Context;

class OverlayState : angle::NonCopyable
{
  public:
    OverlayState();
    ~OverlayState();

    size_t getWidgetCoordinatesBufferSize() const;
    size_t getTextWidgetsBufferSize() const;
    size_t getGraphWidgetsBufferSize() const;

    const uint8_t *getFontData() const;
    void fillWidgetData(const gl::Extents &imageExtents,
                        uint8_t *textData,
                        uint8_t *graphData,
                        uint32_t *activeTextWidgetCountOut,
                        uint32_t *activeGraphWidgetCountOut) const;

    uint32_t getEnabledWidgetCount() const { return mEnabledWidgetCount; }

  private:
    friend class Overlay;

    uint32_t mEnabledWidgetCount;

    angle::PackedEnumMap<WidgetId, std::unique_ptr<overlay::Widget>> mOverlayWidgets;
};

class Overlay : angle::NonCopyable
{
  public:
    Overlay(rx::GLImplFactory *implFactory);
    ~Overlay();

    void init();
    void destroy(const gl::Context *context);

    void onSwap() const;

    overlay::Text *getTextWidget(WidgetId id) const
    {
        return getWidgetAs<overlay::Text, WidgetType::Text>(id);
    }
    overlay::Count *getCountWidget(WidgetId id) const
    {
        return getWidgetAs<overlay::Count, WidgetType::Count>(id);
    }
    overlay::PerSecond *getPerSecondWidget(WidgetId id) const
    {
        return getWidgetAs<overlay::PerSecond, WidgetType::PerSecond>(id);
    }
    overlay::RunningGraph *getRunningGraphWidget(WidgetId id) const
    {
        return getWidgetAs<overlay::RunningGraph, WidgetType::RunningGraph>(id);
    }
    overlay::RunningHistogram *getRunningHistogramWidget(WidgetId id) const
    {
        return getWidgetAs<overlay::RunningHistogram, WidgetType::RunningHistogram>(id);
    }

    rx::OverlayImpl *getImplementation() const { return mImplementation.get(); }

    bool isEnabled() const
    {
        return mImplementation != nullptr && mState.getEnabledWidgetCount() > 0;
    }

  private:
    template <typename Widget, WidgetType Type>
    Widget *getWidgetAs(WidgetId id) const
    {
        ASSERT(mState.mOverlayWidgets[id] != nullptr);
        ASSERT(mState.mOverlayWidgets[id]->type == Type);
        return rx::GetAs<Widget>(mState.mOverlayWidgets[id].get());
    }
    void initOverlayWidgets();
    void enableOverlayWidgetsFromEnvironment();

    // Time tracking for PerSecond items.
    mutable double mLastPerSecondUpdate;

    OverlayState mState;
    std::unique_ptr<rx::OverlayImpl> mImplementation;
};

class MockOverlay
{
  public:
    MockOverlay(rx::GLImplFactory *implFactory);
    ~MockOverlay();

    void init() {}
    void destroy(const Context *context) {}

    void onSwap() const {}

    const overlay::Mock *getTextWidget(WidgetId id) const { return &mMock; }
    const overlay::Mock *getCountWidget(WidgetId id) const { return &mMock; }
    const overlay::Mock *getPerSecondWidget(WidgetId id) const { return &mMock; }
    const overlay::Mock *getRunningGraphWidget(WidgetId id) const { return &mMock; }
    const overlay::Mock *getRunningHistogramWidget(WidgetId id) const { return &mMock; }

    bool isEnabled() const { return false; }

  private:
    overlay::Mock mMock;
};

#if ANGLE_ENABLE_OVERLAY
using OverlayType            = Overlay;
using CountWidget            = overlay::Count;
using PerSecondWidget        = overlay::PerSecond;
using RunningGraphWidget     = overlay::RunningGraph;
using RunningHistogramWidget = overlay::RunningHistogram;
using TextWidget             = overlay::Text;
#else   // !ANGLE_ENABLE_OVERLAY
using OverlayType            = MockOverlay;
using CountWidget            = const overlay::Mock;
using PerSecondWidget        = const overlay::Mock;
using RunningGraphWidget     = const overlay::Mock;
using RunningHistogramWidget = const overlay::Mock;
using TextWidget             = const overlay::Mock;
#endif  // ANGLE_ENABLE_OVERLAY

}  // namespace gl

#endif  // LIBANGLE_OVERLAY_H_

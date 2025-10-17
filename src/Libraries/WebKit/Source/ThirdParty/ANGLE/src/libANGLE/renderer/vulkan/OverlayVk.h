/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 2, 2025.
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
// OverlayVk.h:
//    Defines the OverlayVk class that does the actual rendering of the overlay.
//

#ifndef LIBANGLE_RENDERER_VULKAN_OVERLAYVK_H_
#define LIBANGLE_RENDERER_VULKAN_OVERLAYVK_H_

#include "common/angleutils.h"
#include "libANGLE/Overlay.h"
#include "libANGLE/renderer/OverlayImpl.h"
#include "libANGLE/renderer/vulkan/vk_helpers.h"

namespace rx
{
class ContextVk;

class OverlayVk : public OverlayImpl
{
  public:
    OverlayVk(const gl::OverlayState &state);
    ~OverlayVk() override;

    void onDestroy(const gl::Context *context) override;

    angle::Result onPresent(ContextVk *contextVk,
                            vk::ImageHelper *imageToPresent,
                            const vk::ImageView *imageToPresentView,
                            bool is90DegreeRotation);

    uint32_t getEnabledWidgetCount() const { return mState.getEnabledWidgetCount(); }

  private:
    angle::Result createFont(ContextVk *contextVk);

    vk::ImageHelper mFontImage;
    vk::ImageView mFontImageView;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_VULKAN_OVERLAYVK_H_

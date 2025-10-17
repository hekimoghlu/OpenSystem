/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 1, 2022.
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

#include <WebCore/FloatSize.h>
#include <wtf/RefPtr.h>

namespace WebCore {
class RenderImage;
class ShareableBitmap;
}

namespace WebKit {

enum class UseSnapshotForTransparentImages : bool { No, Yes };
enum class AllowAnimatedImages : bool { No, Yes };

struct CreateShareableBitmapFromImageOptions {
    std::optional<WebCore::FloatSize> screenSizeInPixels;
    AllowAnimatedImages allowAnimatedImages { AllowAnimatedImages::Yes };
    UseSnapshotForTransparentImages useSnapshotForTransparentImages { UseSnapshotForTransparentImages::No };
};

RefPtr<WebCore::ShareableBitmap> createShareableBitmap(WebCore::RenderImage&, CreateShareableBitmapFromImageOptions&& = { });

};

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 21, 2022.
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

#if PLATFORM(COCOA)
OBJC_CLASS CALayer;
using PlatformLayer = CALayer;
#elif USE(COORDINATED_GRAPHICS)
namespace WebCore {
class CoordinatedPlatformLayerBufferProxy;
};
using PlatformLayer = WebCore::CoordinatedPlatformLayerBufferProxy;
#elif USE(TEXTURE_MAPPER)
namespace WebCore {
class TextureMapperPlatformLayer;
};
using PlatformLayer = WebCore::TextureMapperPlatformLayer;
#else
using PlatformLayer = void*;
#endif

#if PLATFORM(COCOA)
#include <wtf/RetainPtr.h>
using PlatformLayerContainer = RetainPtr<PlatformLayer>;
#elif USE(TEXTURE_MAPPER) && !USE(COORDINATED_GRAPHICS)
using PlatformLayerContainer = std::unique_ptr<PlatformLayer>;
#else
#include <wtf/RefPtr.h>
using PlatformLayerContainer = RefPtr<PlatformLayer>;
#endif

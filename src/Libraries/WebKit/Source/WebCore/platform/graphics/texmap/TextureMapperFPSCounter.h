/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 22, 2024.
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
#ifndef TextureMapperFPSCounter_h
#define TextureMapperFPSCounter_h

#include "FloatPoint.h"
#include "TransformationMatrix.h"
#include <wtf/MonotonicTime.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class TextureMapper;

class TextureMapperFPSCounter {
    WTF_MAKE_TZONE_ALLOCATED(TextureMapperFPSCounter);
    WTF_MAKE_NONCOPYABLE(TextureMapperFPSCounter);
public:
    WEBCORE_EXPORT TextureMapperFPSCounter();
    WEBCORE_EXPORT void updateFPSAndDisplay(TextureMapper&, const FloatPoint& = FloatPoint::zero(), const TransformationMatrix& = TransformationMatrix());
    bool isActive() const { return m_isShowingFPS; }

private:
    bool m_isShowingFPS;
    Seconds m_fpsInterval;
    MonotonicTime m_fpsTimestamp;
    int m_lastFPS;
    int m_frameCount;
};

} // namespace WebCore

#endif // TextureMapperFPSCounter_h

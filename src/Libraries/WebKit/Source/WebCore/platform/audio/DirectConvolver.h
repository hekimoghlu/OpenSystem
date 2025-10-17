/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 25, 2024.
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
#ifndef DirectConvolver_h
#define DirectConvolver_h

#include "AudioArray.h"
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class DirectConvolver final {
    WTF_MAKE_TZONE_ALLOCATED(DirectConvolver);
    WTF_MAKE_NONCOPYABLE(DirectConvolver);
public:
    explicit DirectConvolver(size_t inputBlockSize);

    void process(AudioFloatArray* convolutionKernel, std::span<const float> source, std::span<float> destination);

    void reset();

private:
    size_t m_inputBlockSize;

    AudioFloatArray m_buffer;
};

} // namespace WebCore

#endif // DirectConvolver_h

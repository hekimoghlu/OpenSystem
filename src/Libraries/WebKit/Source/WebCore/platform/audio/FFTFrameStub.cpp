/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 28, 2024.
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
// FFTFrame stub implementation to avoid link errors during bringup

#include "config.h"

#if ENABLE(WEB_AUDIO)

#if !OS(DARWIN) && !USE(GSTREAMER)

#include "FFTFrame.h"

namespace WebCore {

// Normal constructor: allocates for a given fftSize.
FFTFrame::FFTFrame(unsigned /*fftSize*/)
    : m_FFTSize(0)
    , m_log2FFTSize(0)
{
    ASSERT_NOT_REACHED();
}

// Creates a blank/empty frame (interpolate() must later be called).
FFTFrame::FFTFrame()
    : m_FFTSize(0)
    , m_log2FFTSize(0)
{
    ASSERT_NOT_REACHED();
}

// Copy constructor.
FFTFrame::FFTFrame(const FFTFrame& frame)
    : m_FFTSize(frame.m_FFTSize)
    , m_log2FFTSize(frame.m_log2FFTSize)
{
    ASSERT_NOT_REACHED();
}

FFTFrame::~FFTFrame()
{
    ASSERT_NOT_REACHED();
}

void FFTFrame::doFFT(std::span<const float> data)
{
    ASSERT_NOT_REACHED();
}

void FFTFrame::doInverseFFT(std::span<float> data)
{
    ASSERT_NOT_REACHED();
}

void FFTFrame::initialize()
{
}

} // namespace WebCore

#endif // !OS(DARWIN) && !USE(GSTREAMER)

#endif // ENABLE(WEB_AUDIO)

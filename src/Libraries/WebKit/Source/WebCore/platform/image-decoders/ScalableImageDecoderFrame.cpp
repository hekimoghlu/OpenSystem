/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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
#include "config.h"
#include "ScalableImageDecoderFrame.h"

#include <wtf/NeverDestroyed.h>

namespace WebCore {

ScalableImageDecoderFrame::ScalableImageDecoderFrame()
{
}

ScalableImageDecoderFrame::~ScalableImageDecoderFrame()
{
}

ScalableImageDecoderFrame& ScalableImageDecoderFrame::operator=(const ScalableImageDecoderFrame& other)
{
    if (this == &other)
        return *this;

    m_decodingStatus = other.m_decodingStatus;

    if (other.backingStore())
        initialize(*other.backingStore());
    else
        m_backingStore = nullptr;
    m_disposalMethod = other.m_disposalMethod;

    m_orientation = other.m_orientation;
    m_duration = other.m_duration;
    m_hasAlpha = other.m_hasAlpha;
    return *this;
}

void ScalableImageDecoderFrame::setDecodingStatus(DecodingStatus decodingStatus)
{
    ASSERT(decodingStatus != DecodingStatus::Decoding);
    m_decodingStatus = decodingStatus;
}

DecodingStatus ScalableImageDecoderFrame::decodingStatus() const
{
    ASSERT(m_decodingStatus != DecodingStatus::Decoding);
    return m_decodingStatus;
}

void ScalableImageDecoderFrame::clear()
{
    *this = ScalableImageDecoderFrame();
}

bool ScalableImageDecoderFrame::initialize(const ImageBackingStore& backingStore)
{
    if (&backingStore == this->backingStore())
        return true;

    m_backingStore = ImageBackingStore::create(backingStore);
    return m_backingStore != nullptr;
}

bool ScalableImageDecoderFrame::initialize(const IntSize& size, bool premultiplyAlpha)
{
    if (size.isEmpty())
        return false;

    m_backingStore = ImageBackingStore::create(size, premultiplyAlpha);
    return m_backingStore != nullptr;
}

IntSize ScalableImageDecoderFrame::size() const
{
    if (hasBackingStore())
        return backingStore()->size();
    return { };
}

}

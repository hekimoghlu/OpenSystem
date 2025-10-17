/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 12, 2022.
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
#ifndef MediaDescription_h
#define MediaDescription_h

#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class MediaDescription : public ThreadSafeRefCounted<MediaDescription> {
public:
    explicit MediaDescription(String&& codec)
        : m_codec(WTFMove(codec))
    {
        ASSERT(m_codec.isSafeToSendToAnotherThread());
    }
    virtual ~MediaDescription() = default;

    StringView codec() const { return m_codec; }
    virtual bool isVideo() const = 0;
    virtual bool isAudio() const = 0;
    virtual bool isText() const = 0;
protected:
    const String m_codec;
};

}

#endif

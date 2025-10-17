/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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

#include "MediaSample.h"
#include <JavaScriptCore/Forward.h>
#include <pal/avfoundation/MediaTimeAVFoundation.h>
#include <wtf/Forward.h>
#include <wtf/TypeCasts.h>

using CVPixelBufferRef = struct __CVBuffer*;

namespace WebCore {

class SharedBuffer;
class PixelBuffer;
class VideoFrameCV;

class MediaSampleAVFObjC : public MediaSample {
public:
    static Ref<MediaSampleAVFObjC> create(CMSampleBufferRef sample, TrackID trackID) { return adoptRef(*new MediaSampleAVFObjC(sample, trackID)); }

    MediaTime presentationTime() const override;
    MediaTime decodeTime() const override;
    MediaTime duration() const override;

    TrackID trackID() const override { return m_id; }

    size_t sizeInBytes() const override;
    FloatSize presentationSize() const override;

    SampleFlags flags() const override;
    PlatformSample platformSample() const override;
    PlatformSample::Type platformSampleType() const override { return PlatformSample::CMSampleBufferType; }
    void offsetTimestampsBy(const MediaTime&) override;
    void setTimestamps(const MediaTime&, const MediaTime&) override;
    WEBCORE_EXPORT bool isDivisable() const override;
    Vector<Ref<MediaSampleAVFObjC>> divide();

    WEBCORE_EXPORT std::pair<RefPtr<MediaSample>, RefPtr<MediaSample>> divide(const MediaTime& presentationTime, UseEndTime) override;
    WEBCORE_EXPORT Ref<MediaSample> createNonDisplayingCopy() const override;

    CMSampleBufferRef sampleBuffer() const { return m_sample.get(); }

    bool isHomogeneous() const;
    Vector<Ref<MediaSampleAVFObjC>> divideIntoHomogeneousSamples();

#if ENABLE(ENCRYPTED_MEDIA) && HAVE(AVCONTENTKEYSESSION)
    using KeyIDs = Vector<Ref<SharedBuffer>>;
    void setKeyIDs(KeyIDs&& keyIDs) { m_keyIDs = WTFMove(keyIDs); }
    const KeyIDs& keyIDs() const { return m_keyIDs; }
    KeyIDs& keyIDs() { return m_keyIDs; }
#endif

    static bool isCMSampleBufferNonDisplaying(CMSampleBufferRef);

protected:
    WEBCORE_EXPORT MediaSampleAVFObjC(RetainPtr<CMSampleBufferRef>&&);
    WEBCORE_EXPORT MediaSampleAVFObjC(CMSampleBufferRef);
    WEBCORE_EXPORT MediaSampleAVFObjC(CMSampleBufferRef, TrackID);
    WEBCORE_EXPORT virtual ~MediaSampleAVFObjC();
    
    void commonInit();

    RetainPtr<CMSampleBufferRef> m_sample;
    TrackID m_id;
    
    MediaTime m_presentationTime;
    MediaTime m_decodeTime;
    MediaTime m_duration;

#if ENABLE(ENCRYPTED_MEDIA) && HAVE(AVCONTENTKEYSESSION)
    Vector<Ref<SharedBuffer>> m_keyIDs;
#endif
};

} // namespace WebCore

namespace WTF {

template<typename Type> struct LogArgument;
template <>
struct LogArgument<WebCore::MediaSampleAVFObjC> {
    static String toString(const WebCore::MediaSampleAVFObjC& sample)
    {
        return sample.toJSONString();
    }
};

} // namespace WTF

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::MediaSampleAVFObjC)
static bool isType(const WebCore::MediaSample& sample) { return sample.platformSampleType() == WebCore::PlatformSample::CMSampleBufferType; }
SPECIALIZE_TYPE_TRAITS_END()

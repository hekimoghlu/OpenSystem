/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 20, 2022.
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

#if USE(GSTREAMER)

#include "GStreamerCommon.h"
#include "MediaPlayer.h"
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/Nonmovable.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class MediaPlayerPrivateGStreamer;

enum class ElementRuntimeCharacteristics : uint8_t {
    IsMediaStream = 1 << 0,
    HasVideo = 1 << 1,
    HasAudio = 1 << 2,
    IsLiveStream = 1 << 3,
};

class GStreamerQuirkBase {
    WTF_MAKE_TZONE_ALLOCATED(GStreamerQuirkBase);

public:
    GStreamerQuirkBase() = default;
    virtual ~GStreamerQuirkBase() = default;

    virtual const ASCIILiteral identifier() const = 0;

    // Interface of classes supplied to MediaPlayerPrivateGStreamer to store values that the quirks will need for their job.
    class GStreamerQuirkState {
        WTF_MAKE_FAST_ALLOCATED;
        // Prevent accidental https://en.wikipedia.org/wiki/Object_slicing.
        WTF_MAKE_NONCOPYABLE(GStreamerQuirkState);
        WTF_MAKE_NONMOVABLE(GStreamerQuirkState);
    public:
        GStreamerQuirkState()
        {
        }
        virtual ~GStreamerQuirkState() = default;
    };
};

class GStreamerQuirk : public GStreamerQuirkBase {
    WTF_MAKE_TZONE_ALLOCATED(GStreamerQuirk);
public:
    GStreamerQuirk() = default;
    virtual ~GStreamerQuirk() = default;

    virtual bool isPlatformSupported() const { return true; }
    virtual GstElement* createAudioSink() { return nullptr; }
    virtual GstElement* createWebAudioSink() { return nullptr; }
    virtual void configureElement(GstElement*, const OptionSet<ElementRuntimeCharacteristics>&) { }
    virtual std::optional<bool> isHardwareAccelerated(GstElementFactory*) { return std::nullopt; }
    virtual std::optional<GstElementFactoryListType> audioVideoDecoderFactoryListType() const { return std::nullopt; }
    virtual Vector<String> disallowedWebAudioDecoders() const { return { }; }
    virtual unsigned getAdditionalPlaybinFlags() const { return 0; }
    virtual bool shouldParseIncomingLibWebRTCBitStream() const { return true; }

    virtual bool needsBufferingPercentageCorrection() const { return false; }
    // Returns name of the queried GstElement, or nullptr if no element was queried.
    virtual ASCIILiteral queryBufferingPercentage(MediaPlayerPrivateGStreamer*, const GRefPtr<GstQuery>&) const { return nullptr; }
    virtual int correctBufferingPercentage(MediaPlayerPrivateGStreamer*, int originalBufferingPercentage, GstBufferingMode) const { return originalBufferingPercentage; }
    virtual void resetBufferingPercentage(MediaPlayerPrivateGStreamer*, int) const { };
    virtual void setupBufferingPercentageCorrection(MediaPlayerPrivateGStreamer*, GstState, GstState, GRefPtr<GstElement>&&) const { }

    // Subclass must return true if it wants to override the default behaviour of sibling platforms.
    virtual bool processWebAudioSilentBuffer(GstBuffer* buffer) const
    {
        GST_BUFFER_FLAG_SET(buffer, GST_BUFFER_FLAG_GAP);
        GST_BUFFER_FLAG_SET(buffer, GST_BUFFER_FLAG_DROPPABLE);
        return false;
    }
};

class GStreamerHolePunchQuirk : public GStreamerQuirkBase {
    WTF_MAKE_TZONE_ALLOCATED(GStreamerHolePunchQuirk);
public:
    GStreamerHolePunchQuirk() = default;
    virtual ~GStreamerHolePunchQuirk() = default;

    virtual GstElement* createHolePunchVideoSink(bool, const MediaPlayer*) { return nullptr; }
    virtual bool setHolePunchVideoRectangle(GstElement*, const IntRect&) { return false; }
    virtual bool requiresClockSynchronization() const { return true; }
};

class GStreamerQuirksManager : public RefCounted<GStreamerQuirksManager> {
    friend NeverDestroyed<GStreamerQuirksManager>;
    WTF_MAKE_TZONE_ALLOCATED(GStreamerQuirksManager);

public:
    static GStreamerQuirksManager& singleton();

    static RefPtr<GStreamerQuirksManager> createForTesting()
    {
        return adoptRef(*new GStreamerQuirksManager(true, false));
    }

    bool isEnabled() const;

    GstElement* createAudioSink();
    GstElement* createWebAudioSink();
    void configureElement(GstElement*, OptionSet<ElementRuntimeCharacteristics>&&);
    std::optional<bool> isHardwareAccelerated(GstElementFactory*) const;
    GstElementFactoryListType audioVideoDecoderFactoryListType() const;
    Vector<String> disallowedWebAudioDecoders() const;

    bool supportsVideoHolePunchRendering() const;
    GstElement* createHolePunchVideoSink(bool isLegacyPlaybin, const MediaPlayer*);
    void setHolePunchVideoRectangle(GstElement*, const IntRect&);
    bool sinksRequireClockSynchronization() const;

    void setHolePunchEnabledForTesting(bool);

    unsigned getAdditionalPlaybinFlags() const;

    bool shouldParseIncomingLibWebRTCBitStream() const;

    bool needsBufferingPercentageCorrection() const;
    // Returns name of the queried GstElement, or nullptr if no element was queried.
    ASCIILiteral queryBufferingPercentage(MediaPlayerPrivateGStreamer*, const GRefPtr<GstQuery>&) const;
    int correctBufferingPercentage(MediaPlayerPrivateGStreamer*, int originalBufferingPercentage, GstBufferingMode) const;
    void resetBufferingPercentage(MediaPlayerPrivateGStreamer*, int bufferingPercentage) const;
    void setupBufferingPercentageCorrection(MediaPlayerPrivateGStreamer*, GstState currentState, GstState newState, GRefPtr<GstElement>&&) const;

    void processWebAudioSilentBuffer(GstBuffer*) const;
private:
    GStreamerQuirksManager(bool, bool);

    Vector<std::unique_ptr<GStreamerQuirk>> m_quirks;
    std::unique_ptr<GStreamerHolePunchQuirk> m_holePunchQuirk;
    bool m_isForTesting { false };
};

} // namespace WebCore

#endif // USE(GSTREAMER)

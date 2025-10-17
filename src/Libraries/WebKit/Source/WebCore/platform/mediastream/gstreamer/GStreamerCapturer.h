/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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

#if ENABLE(MEDIA_STREAM) && USE(GSTREAMER)

#include "GStreamerCaptureDevice.h"
#include "GStreamerCommon.h"

#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/WeakHashSet.h>

namespace WebCore {
class GStreamerCapturerObserver;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::GStreamerCapturerObserver> : std::true_type { };
}

namespace WebCore {

class GStreamerCapturerObserver : public CanMakeWeakPtr<GStreamerCapturerObserver> {
public:
    virtual ~GStreamerCapturerObserver();

    virtual void sourceCapsChanged(const GstCaps*) { }
    virtual void captureEnded() { }
};

class GStreamerCapturer : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<GStreamerCapturer> {
public:
    GStreamerCapturer(GStreamerCaptureDevice&&, GRefPtr<GstCaps>&&);
    GStreamerCapturer(const char* sourceFactory, GRefPtr<GstCaps>&&, CaptureDevice::DeviceType);
    virtual ~GStreamerCapturer();

    void tearDown(bool disconnectSignals = true);

    void addObserver(GStreamerCapturerObserver&);
    void removeObserver(GStreamerCapturerObserver&);
    void forEachObserver(const Function<void(GStreamerCapturerObserver&)>&);

    void setupPipeline();
    void start();
    void stop();
    bool isStopped() const;
    WARN_UNUSED_RETURN GRefPtr<GstCaps> caps();

    std::pair<GstClockTime, GstClockTime> queryLatency();

    GstElement* makeElement(const char* factoryName);
    virtual GstElement* createSource();
    GstElement* source() { return m_src.get();  }
    virtual const char* name() = 0;

    GstElement* sink() const { return m_sink.get(); }

    GstElement* pipeline() const { return m_pipeline.get(); }
    virtual GstElement* createConverter() = 0;

    bool isInterrupted() const;
    void setInterrupted(bool);

    CaptureDevice::DeviceType deviceType() const { return m_deviceType; }
    const String& devicePersistentId() const { return m_device ? m_device->persistentId() : emptyString(); }

    void stopDevice(bool disconnectSignals);

protected:
    GRefPtr<GstElement> m_sink;
    GRefPtr<GstElement> m_src;
    GRefPtr<GstElement> m_valve;
    GRefPtr<GstElement> m_capsfilter;
    std::optional<GStreamerCaptureDevice> m_device { };
    GRefPtr<GstCaps> m_caps;
    GRefPtr<GstElement> m_pipeline;
    const char* m_sourceFactory;

private:
    CaptureDevice::DeviceType m_deviceType;
    WeakHashSet<GStreamerCapturerObserver> m_observers;
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM) && USE(GSTREAMER)

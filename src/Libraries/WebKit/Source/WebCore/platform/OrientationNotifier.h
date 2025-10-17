/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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

#include "IntDegrees.h"
#include <wtf/CheckedRef.h>
#include <wtf/Vector.h>

namespace WebCore {

enum class VideoFrameRotation : uint16_t;

class OrientationNotifier final : public CanMakeCheckedPtr<OrientationNotifier> {
    WTF_MAKE_TZONE_ALLOCATED(OrientationNotifier);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(OrientationNotifier);
public:
    explicit OrientationNotifier(IntDegrees orientation) { m_orientation = orientation; }
    ~OrientationNotifier();

    void orientationChanged(IntDegrees orientation);
    void rotationAngleForCaptureDeviceChanged(const String&, VideoFrameRotation);

    class Observer {
    public:
        virtual ~Observer();
        virtual void orientationChanged(IntDegrees orientation) = 0;
        virtual void rotationAngleForHorizonLevelDisplayChanged(const String&, VideoFrameRotation) { }
        void setNotifier(OrientationNotifier*);

    private:
        OrientationNotifier* m_notifier { nullptr };
    };

    void addObserver(Observer&);
    void removeObserver(Observer&);
    IntDegrees orientation() const { return m_orientation; }

private:
    Vector<std::reference_wrapper<Observer>> m_observers;
    IntDegrees m_orientation;
};

inline OrientationNotifier::~OrientationNotifier()
{
    for (Observer& observer : m_observers)
        observer.setNotifier(nullptr);
}

inline OrientationNotifier::Observer::~Observer()
{
    if (m_notifier)
        m_notifier->removeObserver(*this);
}

inline void OrientationNotifier::Observer::setNotifier(OrientationNotifier* notifier)
{
    if (m_notifier == notifier)
        return;

    if (m_notifier && notifier)
        m_notifier->removeObserver(*this);

    ASSERT(!m_notifier || !notifier);
    m_notifier = notifier;
}

inline void OrientationNotifier::orientationChanged(IntDegrees orientation)
{
    m_orientation = orientation;
    for (Observer& observer : m_observers)
        observer.orientationChanged(orientation);
}

inline void OrientationNotifier::rotationAngleForCaptureDeviceChanged(const String& devicePersistentId, VideoFrameRotation orientation)
{
    for (Observer& observer : m_observers)
        observer.rotationAngleForHorizonLevelDisplayChanged(devicePersistentId, orientation);
}

inline void OrientationNotifier::addObserver(Observer& observer)
{
    m_observers.append(observer);
    observer.setNotifier(this);
}

inline void OrientationNotifier::removeObserver(Observer& observer)
{
    m_observers.removeFirstMatching([&observer](auto item) {
        if (&observer != &item.get())
            return false;
        observer.setNotifier(nullptr);
        return true;
    });
}

} // namespace WebCore

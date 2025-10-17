/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 20, 2023.
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

#if ENABLE(MEDIA_STREAM)
#include "RealtimeMediaSourceSettings.h"

#include <wtf/NeverDestroyed.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

RealtimeMediaSourceSettings RealtimeMediaSourceSettings::isolatedCopy() const
{
    return { m_width, m_height , m_frameRate, m_facingMode, m_volume , m_sampleRate, m_sampleSize, m_echoCancellation, m_deviceId.isolatedCopy(), m_groupId.isolatedCopy(), m_label.isolatedCopy(), m_displaySurface, m_logicalSurface, m_whiteBalanceMode, m_zoom, m_torch, m_backgroundBlur, m_powerEfficient, RealtimeMediaSourceSupportedConstraints { m_supportedConstraints } };
}

VideoFacingMode RealtimeMediaSourceSettings::videoFacingModeEnum(const String& mode)
{
    if (mode == "user"_s)
        return VideoFacingMode::User;
    if (mode == "environment"_s)
        return VideoFacingMode::Environment;
    if (mode == "left"_s)
        return VideoFacingMode::Left;
    if (mode == "right"_s)
        return VideoFacingMode::Right;

    return VideoFacingMode::Unknown;
}

String RealtimeMediaSourceSettings::convertFlagsToString(const OptionSet<RealtimeMediaSourceSettings::Flag> flags)
{
    StringBuilder builder;

    builder.append("[ "_s);
    for (auto flag : flags) {
        if (!builder.isEmpty())
            builder.append(", "_s);

        switch (flag) {
        case RealtimeMediaSourceSettings::Width:
            builder.append("Width"_s);
            break;
        case RealtimeMediaSourceSettings::Height:
            builder.append("Height"_s);
            break;
        case RealtimeMediaSourceSettings::FrameRate:
            builder.append("FrameRate"_s);
            break;
        case RealtimeMediaSourceSettings::FacingMode:
            builder.append("FacingMode"_s);
            break;
        case RealtimeMediaSourceSettings::Volume:
            builder.append("Volume"_s);
            break;
        case RealtimeMediaSourceSettings::SampleRate:
            builder.append("SampleRate"_s);
            break;
        case RealtimeMediaSourceSettings::SampleSize:
            builder.append("SampleSize"_s);
            break;
        case RealtimeMediaSourceSettings::EchoCancellation:
            builder.append("EchoCancellation"_s);
            break;
        case RealtimeMediaSourceSettings::DeviceId:
            builder.append("DeviceId"_s);
            break;
        case RealtimeMediaSourceSettings::GroupId:
            builder.append("GroupId"_s);
            break;
        case RealtimeMediaSourceSettings::Label:
            builder.append("Label"_s);
            break;
        case RealtimeMediaSourceSettings::DisplaySurface:
            builder.append("DisplaySurface"_s);
            break;
        case RealtimeMediaSourceSettings::LogicalSurface:
            builder.append("LogicalSurface"_s);
            break;
        case RealtimeMediaSourceSettings::WhiteBalanceMode:
            builder.append("WhiteBalanceMode"_s);
            break;
        case RealtimeMediaSourceSettings::Zoom:
            builder.append("Zoom"_s);
            break;
        case RealtimeMediaSourceSettings::Torch:
            builder.append("Torch"_s);
            break;
        case RealtimeMediaSourceSettings::BackgroundBlur:
            builder.append("BackgroundBlur"_s);
            break;
        case RealtimeMediaSourceSettings::PowerEfficient:
            builder.append("PowerEfficient"_s);
            break;
        }
    }
    builder.append(" ]"_s);

    return builder.toString();
}

OptionSet<RealtimeMediaSourceSettings::Flag> RealtimeMediaSourceSettings::difference(const RealtimeMediaSourceSettings& that) const
{
    OptionSet<RealtimeMediaSourceSettings::Flag> difference;

    if (width() != that.width())
        difference.add(RealtimeMediaSourceSettings::Width);
    if (height() != that.height())
        difference.add(RealtimeMediaSourceSettings::Height);
    if (frameRate() != that.frameRate())
        difference.add(RealtimeMediaSourceSettings::FrameRate);
    if (facingMode() != that.facingMode())
        difference.add(RealtimeMediaSourceSettings::FacingMode);
    if (volume() != that.volume())
        difference.add(RealtimeMediaSourceSettings::Volume);
    if (sampleRate() != that.sampleRate())
        difference.add(RealtimeMediaSourceSettings::SampleRate);
    if (sampleSize() != that.sampleSize())
        difference.add(RealtimeMediaSourceSettings::SampleSize);
    if (echoCancellation() != that.echoCancellation())
        difference.add(RealtimeMediaSourceSettings::EchoCancellation);
    if (deviceId() != that.deviceId())
        difference.add(RealtimeMediaSourceSettings::DeviceId);
    if (groupId() != that.groupId())
        difference.add(RealtimeMediaSourceSettings::GroupId);
    if (label() != that.label())
        difference.add(RealtimeMediaSourceSettings::Label);
    if (displaySurface() != that.displaySurface())
        difference.add(RealtimeMediaSourceSettings::DisplaySurface);
    if (logicalSurface() != that.logicalSurface())
        difference.add(RealtimeMediaSourceSettings::LogicalSurface);
    if (whiteBalanceMode() != that.whiteBalanceMode())
        difference.add(RealtimeMediaSourceSettings::WhiteBalanceMode);
    if (zoom() != that.zoom())
        difference.add(RealtimeMediaSourceSettings::Zoom);
    if (torch() != that.torch())
        difference.add(RealtimeMediaSourceSettings::Torch);
    if (backgroundBlur() != that.backgroundBlur())
        difference.add(RealtimeMediaSourceSettings::BackgroundBlur);
    if (powerEfficient() != that.powerEfficient())
        difference.add(RealtimeMediaSourceSettings::PowerEfficient);

    return difference;
}

String convertEnumerationToString(VideoFacingMode enumerationValue)
{
    static const std::array<NeverDestroyed<String>, 5> values {
        MAKE_STATIC_STRING_IMPL("unknown"),
        MAKE_STATIC_STRING_IMPL("user"),
        MAKE_STATIC_STRING_IMPL("environment"),
        MAKE_STATIC_STRING_IMPL("left"),
        MAKE_STATIC_STRING_IMPL("right"),
    };
    static_assert(static_cast<size_t>(VideoFacingMode::Unknown) == 0, "VideoFacingMode::Unknown is not 0 as expected");
    static_assert(static_cast<size_t>(VideoFacingMode::User) == 1, "VideoFacingMode::User is not 1 as expected");
    static_assert(static_cast<size_t>(VideoFacingMode::Environment) == 2, "VideoFacingMode::Environment is not 2 as expected");
    static_assert(static_cast<size_t>(VideoFacingMode::Left) == 3, "VideoFacingMode::Left is not 3 as expected");
    static_assert(static_cast<size_t>(VideoFacingMode::Right) == 4, "VideoFacingMode::Right is not 4 as expected");
    ASSERT(static_cast<size_t>(enumerationValue) < std::size(values));
    return values[static_cast<size_t>(enumerationValue)];
}

String RealtimeMediaSourceSettings::displaySurface(DisplaySurfaceType surface)
{
    static const std::array<NeverDestroyed<String>, 5> values {
        MAKE_STATIC_STRING_IMPL("monitor"),
        MAKE_STATIC_STRING_IMPL("window"),
        MAKE_STATIC_STRING_IMPL("application"),
        MAKE_STATIC_STRING_IMPL("browser"),
        MAKE_STATIC_STRING_IMPL("invalid"),
    };

    static_assert(static_cast<size_t>(DisplaySurfaceType::Monitor) == 0, "RealtimeMediaSourceSettings::DisplaySurface::Monitor is not 0 as expected");
    static_assert(static_cast<size_t>(DisplaySurfaceType::Window) == 1, "RealtimeMediaSourceSettings::DisplaySurface::Window is not 1 as expected");
    static_assert(static_cast<size_t>(DisplaySurfaceType::Application) == 2, "RealtimeMediaSourceSettings::DisplaySurface::Application is not 0 as expected");
    static_assert(static_cast<size_t>(DisplaySurfaceType::Browser) == 3, "RealtimeMediaSourceSettings::DisplaySurface::Browser is not 1 as expected");
    static_assert(static_cast<size_t>(DisplaySurfaceType::Invalid) == 4, "RealtimeMediaSourceSettings::DisplaySurface::Invalid is not 0 as expected");
    ASSERT(static_cast<size_t>(surface) < std::size(values));
    return values[static_cast<size_t>(surface)];
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)

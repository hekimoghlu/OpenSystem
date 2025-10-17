/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 24, 2021.
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
#include "MediaTrackConstraints.h"

#if ENABLE(MEDIA_STREAM)

#include "MediaConstraints.h"

namespace WebCore {

enum class ConstraintSetType { Mandatory, Advanced };

static void set(MediaTrackConstraintSetMap& map, ConstraintSetType setType, MediaConstraintType type, const ConstrainLong& value)
{
    IntConstraint constraint;
    WTF::switchOn(value,
        [&] (int integer) {
            if (setType == ConstraintSetType::Mandatory)
                constraint.setIdeal(integer);
            else
                constraint.setExact(integer);
        },
        [&] (const ConstrainLongRange& range) {
            if (range.min)
                constraint.setMin(range.min.value());
            if (range.max)
                constraint.setMax(range.max.value());
            if (range.exact)
                constraint.setExact(range.exact.value());
            if (range.ideal)
                constraint.setIdeal(range.ideal.value());
        }
    );
    map.set(type, WTFMove(constraint));
}

static void set(MediaTrackConstraintSetMap& map, ConstraintSetType setType, MediaConstraintType type, const ConstrainDouble& value)
{
    DoubleConstraint constraint;
    WTF::switchOn(value,
        [&] (double number) {
            if (setType == ConstraintSetType::Mandatory)
                constraint.setIdeal(number);
            else
                constraint.setExact(number);
        },
        [&] (const ConstrainDoubleRange& range) {
            if (range.min)
                constraint.setMin(range.min.value());
            if (range.max)
                constraint.setMax(range.max.value());
            if (range.exact)
                constraint.setExact(range.exact.value());
            if (range.ideal)
                constraint.setIdeal(range.ideal.value());
        }
    );
    map.set(type, WTFMove(constraint));
}

static void set(MediaTrackConstraintSetMap& map, ConstraintSetType setType, MediaConstraintType type, const ConstrainBoolean& value)
{
    BooleanConstraint constraint;
    WTF::switchOn(value,
        [&] (bool boolean) {
            if (setType == ConstraintSetType::Mandatory)
                constraint.setIdeal(boolean);
            else
                constraint.setExact(boolean);
        },
        [&] (const ConstrainBooleanParameters& parameters) {
            if (parameters.exact)
                constraint.setExact(parameters.exact.value());
            if (parameters.ideal)
                constraint.setIdeal(parameters.ideal.value());
        }
    );
    map.set(type, WTFMove(constraint));
}

static void set(MediaTrackConstraintSetMap& map, ConstraintSetType setType, MediaConstraintType type, const ConstrainDOMString& value)
{
    StringConstraint constraint;
    WTF::switchOn(value,
        [&] (const String& string) {
            if (setType == ConstraintSetType::Mandatory)
                constraint.appendIdeal(string);
            else
                constraint.appendExact(string);
        },
        [&] (const Vector<String>& vector) {
            if (setType == ConstraintSetType::Mandatory) {
                for (auto& string : vector)
                    constraint.appendIdeal(string);
            } else {
                for (auto& string : vector)
                    constraint.appendExact(string);
            }
        },
        [&] (const ConstrainDOMStringParameters& parameters) {
            if (parameters.exact) {
                WTF::switchOn(parameters.exact.value(),
                    [&] (const String& string) {
                        constraint.appendExact(string);
                    },
                    [&] (const Vector<String>& vector) {
                        for (auto& string : vector)
                            constraint.appendExact(string);
                    }
                );
            }
            if (parameters.ideal) {
                WTF::switchOn(parameters.ideal.value(),
                    [&] (const String& string) {
                        constraint.appendIdeal(string);
                    },
                    [&] (const Vector<String>& vector) {
                        for (auto& string : vector)
                            constraint.appendIdeal(string);
                    }
                );
            }
        }
    );
    map.set(type, WTFMove(constraint));
}

template<typename T> static inline void set(MediaTrackConstraintSetMap& map, ConstraintSetType setType, MediaConstraintType type, const std::optional<T>& value)
{
    if (!value)
        return;
    set(map, setType, type, value.value());
}

static MediaTrackConstraintSetMap convertToInternalForm(ConstraintSetType setType, const MediaTrackConstraintSet& constraintSet)
{
    MediaTrackConstraintSetMap result;
    set(result, setType, MediaConstraintType::Width, constraintSet.width);
    set(result, setType, MediaConstraintType::Height, constraintSet.height);
    set(result, setType, MediaConstraintType::AspectRatio, constraintSet.aspectRatio);
    set(result, setType, MediaConstraintType::FrameRate, constraintSet.frameRate);
    set(result, setType, MediaConstraintType::FacingMode, constraintSet.facingMode);
    set(result, setType, MediaConstraintType::Volume, constraintSet.volume);
    set(result, setType, MediaConstraintType::SampleRate, constraintSet.sampleRate);
    set(result, setType, MediaConstraintType::SampleSize, constraintSet.sampleSize);
    set(result, setType, MediaConstraintType::EchoCancellation, constraintSet.echoCancellation);
    // FIXME: add latency
    // FIXME: add channelCount
    set(result, setType, MediaConstraintType::DeviceId, constraintSet.deviceId);
    set(result, setType, MediaConstraintType::GroupId, constraintSet.groupId);
    set(result, setType, MediaConstraintType::DisplaySurface, constraintSet.displaySurface);
    set(result, setType, MediaConstraintType::LogicalSurface, constraintSet.logicalSurface);

    set(result, setType, MediaConstraintType::WhiteBalanceMode, constraintSet.whiteBalanceMode);
    set(result, setType, MediaConstraintType::Zoom, constraintSet.zoom);
    set(result, setType, MediaConstraintType::Torch, constraintSet.torch);

    set(result, setType, MediaConstraintType::BackgroundBlur, constraintSet.backgroundBlur);
    set(result, setType, MediaConstraintType::PowerEfficient, constraintSet.powerEfficient);
    return result;
}

static Vector<MediaTrackConstraintSetMap> convertAdvancedToInternalForm(const Vector<MediaTrackConstraintSet>& vector)
{
    return vector.map([](auto& set) {
        return convertToInternalForm(ConstraintSetType::Advanced, set);
    });
}

static Vector<MediaTrackConstraintSetMap> convertAdvancedToInternalForm(const std::optional<Vector<MediaTrackConstraintSet>>& optionalVector)
{
    if (!optionalVector)
        return { };
    return convertAdvancedToInternalForm(optionalVector.value());
}

MediaConstraints createMediaConstraints(const MediaTrackConstraints& trackConstraints)
{
    MediaConstraints constraints;
    constraints.mandatoryConstraints = convertToInternalForm(ConstraintSetType::Mandatory, trackConstraints);
    constraints.advancedConstraints = convertAdvancedToInternalForm(trackConstraints.advanced);
    constraints.isValid = true;
    return constraints;
}

}

#endif

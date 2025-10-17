/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 18, 2022.
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

#if ENABLE(MEDIA_STREAM)

#include "DoubleRange.h"
#include "LongRange.h"
#include <variant>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct MediaConstraints;

struct ConstrainBooleanParameters {
    std::optional<bool> exact;
    std::optional<bool> ideal;
};

struct ConstrainDOMStringParameters {
    std::optional<std::variant<String, Vector<String>>> exact;
    std::optional<std::variant<String, Vector<String>>> ideal;
};

struct ConstrainDoubleRange : DoubleRange {
    std::optional<double> exact;
    std::optional<double> ideal;
};

struct ConstrainLongRange : LongRange {
    std::optional<int> exact;
    std::optional<int> ideal;
};

using ConstrainBoolean = std::variant<bool, ConstrainBooleanParameters>;
using ConstrainDOMString = std::variant<String, Vector<String>, ConstrainDOMStringParameters>;
using ConstrainDouble = std::variant<double, ConstrainDoubleRange>;
using ConstrainLong = std::variant<int, ConstrainLongRange>;

struct MediaTrackConstraintSet {
    std::optional<ConstrainLong> width;
    std::optional<ConstrainLong> height;
    std::optional<ConstrainDouble> aspectRatio;
    std::optional<ConstrainDouble> frameRate;
    std::optional<ConstrainDOMString> facingMode;
    std::optional<ConstrainDouble> volume;
    std::optional<ConstrainLong> sampleRate;
    std::optional<ConstrainLong> sampleSize;
    std::optional<ConstrainBoolean> echoCancellation;
    std::optional<ConstrainDOMString> deviceId;
    std::optional<ConstrainDOMString> groupId;
    std::optional<ConstrainDOMString> displaySurface;
    std::optional<ConstrainBoolean> logicalSurface;

    std::optional<ConstrainDOMString> whiteBalanceMode;
    std::optional<ConstrainDouble> zoom;
    std::optional<ConstrainBoolean> torch;

    std::optional<ConstrainBoolean> backgroundBlur;
    std::optional<ConstrainBoolean> powerEfficient;
};

struct MediaTrackConstraints : MediaTrackConstraintSet {
    std::optional<Vector<MediaTrackConstraintSet>> advanced;
};

MediaConstraints createMediaConstraints(const MediaTrackConstraints&);

}

#endif

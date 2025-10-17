/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 3, 2022.
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

#include <CoreText/CoreText.h>

namespace WebCore {

struct FontInterrogation {
    FontInterrogation(CTFontRef font)
    {
        bool foundStat = false;
        bool foundTrak = false;
        auto tables = adoptCF(CTFontCopyAvailableTables(font, kCTFontTableOptionNoOptions));
        if (!tables)
            return;
        auto size = CFArrayGetCount(tables.get());
        for (CFIndex i = 0; i < size; ++i) {
            auto tableTag = static_cast<CTFontTableTag>(reinterpret_cast<uintptr_t>(CFArrayGetValueAtIndex(tables.get(), i)));
            switch (tableTag) {
            case kCTFontTableFvar:
                if (variationType == VariationType::NotVariable)
                    variationType = VariationType::TrueTypeGX;
                break;
            case kCTFontTableSTAT:
                foundStat = true;
                variationType = VariationType::OpenType18;
                break;
            case kCTFontTableMorx:
            case kCTFontTableMort:
                aatShaping = true;
                break;
            case kCTFontTableGPOS:
            case kCTFontTableGSUB:
                openTypeShaping = true;
                break;
            case kCTFontTableTrak:
                foundTrak = true;
                break;
            }
        }
        if (foundStat && foundTrak)
            trackingType = TrackingType::Automatic;
        else if (foundTrak)
            trackingType = TrackingType::Manual;
    }

    enum class VariationType : uint8_t { NotVariable, TrueTypeGX, OpenType18, };
    VariationType variationType { VariationType::NotVariable };
    enum class TrackingType : uint8_t { None, Automatic, Manual, };
    TrackingType trackingType { TrackingType::None };
    bool openTypeShaping { false };
    bool aatShaping { false };
};

} // namespace WebCore

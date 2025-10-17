/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 3, 2022.
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
#import "config.h"
#import "AVAssetTrackUtilities.h"

#if ENABLE(VIDEO) && USE(AVFOUNDATION)

#import "FourCC.h"
#import "SystemBattery.h"
#import "VideoToolboxSoftLink.h"
#import <AVFoundation/AVAssetTrack.h>

#import <pal/cf/CoreMediaSoftLink.h>

namespace WebCore {

static Vector<FourCC> contentTypesToCodecs(const Vector<ContentType>& contentTypes)
{
    Vector<FourCC> codecs;
    for (auto& contentType : contentTypes) {
        auto codecStrings = contentType.codecs();
        for (auto& codecString : codecStrings) {
            // https://tools.ietf.org/html/rfc6381
            // Within a 'codecs' parameter value, "." is reserved as a hierarchy delimiter
            if (auto codecIdentifier = FourCC::fromString(StringView(codecString).left(codecString.find('.')).left(4)))
                codecs.append(codecIdentifier.value());
        }
    }
    return codecs;
}

bool codecsMeetHardwareDecodeRequirements(const Vector<FourCC>& codecs, const Vector<ContentType>& contentTypesRequiringHardwareDecode)
{
    static bool hasBattery = systemHasBattery();

    // If the system is exclusively wall-powered, do not require hardware support.
    if (!hasBattery)
        return true;

    // If we can't determine whether a codec has hardware support or not, default to true.
    if (!canLoad_VideoToolbox_VTIsHardwareDecodeSupported())
        return true;

    if (contentTypesRequiringHardwareDecode.isEmpty())
        return true;

    Vector<FourCC> hardwareCodecs = contentTypesToCodecs(contentTypesRequiringHardwareDecode);

    for (auto& codec : codecs) {
        if (hardwareCodecs.contains(codec) && !VTIsHardwareDecodeSupported(codec.value))
            return false;
    }
    return true;
}

bool contentTypeMeetsHardwareDecodeRequirements(const ContentType& contentType, const Vector<ContentType>& contentTypesRequiringHardwareDecode)
{
    Vector<FourCC> codecs = contentTypesToCodecs({ contentType });
    return codecsMeetHardwareDecodeRequirements(codecs, contentTypesRequiringHardwareDecode);
}

bool assetTrackMeetsHardwareDecodeRequirements(AVAssetTrack *track, const Vector<ContentType>& contentTypesRequiringHardwareDecode)
{
    Vector<FourCC> codecs;
    for (NSUInteger i = 0, count = track.formatDescriptions.count; i < count; ++i) {
        CMFormatDescriptionRef description = (__bridge CMFormatDescriptionRef)track.formatDescriptions[i];
        if (PAL::CMFormatDescriptionGetMediaType(description) == kCMMediaType_Video)
            codecs.append(FourCC(PAL::CMFormatDescriptionGetMediaSubType(description)));
    }
    return codecsMeetHardwareDecodeRequirements(codecs, contentTypesRequiringHardwareDecode);
}

}

#endif

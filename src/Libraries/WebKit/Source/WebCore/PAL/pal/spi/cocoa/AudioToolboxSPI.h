/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 30, 2024.
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
#if USE(APPLE_INTERNAL_SDK)

#import <AudioToolbox/AudioComponentPriv.h>
#import <AudioToolbox/AudioFormatPriv.h>
#import <AudioToolbox/Spatialization.h>

#else

static constexpr OSType kAudioFormatProperty_AvailableDecodeChannelLayoutTags = 'adcl';
static constexpr OSType kAudioFormatProperty_VorbisModeInfo = 'vnfo';

struct AudioFormatVorbisModeInfo {
    UInt32 mShortBlockSize;
    UInt32 mLongBlockSize;
    UInt32 mModeCount;
    UInt64 mModeFlags;
};
typedef struct AudioFormatVorbisModeInfo AudioFormatVorbisModeInfo;

enum SpatialAudioSourceID : UInt32;
static constexpr UInt32 kSpatialAudioSource_Multichannel = 'mlti';
static constexpr UInt32 kSpatialAudioSource_MonoOrStereo = 'most';
static constexpr UInt32 kSpatialAudioSource_BinauralForHeadphones = 'binh';
static constexpr UInt32 kSpatialAudioSource_Unknown = '?src';

enum SpatialContentTypeID : UInt32;
static constexpr UInt32 kAudioSpatialContentType_Audiovisual = 'moov';
static constexpr UInt32 kAudioSpatialContentType_AudioOnly = 'soun';

struct SpatialAudioPreferences {
    Boolean prefersHeadTrackedSpatialization;
    Boolean prefersLossyAudioSources;
    Boolean alwaysSpatialize;
    Boolean pad[5];
    UInt32 maxSpatializableChannels;
    UInt32 spatialAudioSourceCount;
    SpatialAudioSourceID spatialAudioSources[3];
};

#endif

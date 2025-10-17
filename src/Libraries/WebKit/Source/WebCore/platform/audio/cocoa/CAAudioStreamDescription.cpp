/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 11, 2023.
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
#include "CAAudioStreamDescription.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CAAudioStreamDescription);

CAAudioStreamDescription::~CAAudioStreamDescription() = default;

CAAudioStreamDescription::CAAudioStreamDescription(const AudioStreamBasicDescription &desc)
    : m_streamDescription(desc)
{
}

CAAudioStreamDescription::CAAudioStreamDescription(double sampleRate, uint32_t numChannels, PCMFormat format, IsInterleaved isInterleaved)
{
    m_streamDescription.mFormatID = kAudioFormatLinearPCM;
    m_streamDescription.mSampleRate = sampleRate;
    m_streamDescription.mFramesPerPacket = 1;
    m_streamDescription.mChannelsPerFrame = numChannels;
    m_streamDescription.mBytesPerFrame = 0;
    m_streamDescription.mBytesPerPacket = 0;
    m_streamDescription.mFormatFlags = static_cast<AudioFormatFlags>(kAudioFormatFlagsNativeEndian) | static_cast<AudioFormatFlags>(kAudioFormatFlagIsPacked);
    m_streamDescription.mReserved = 0;

    int wordsize;
    switch (format) {
    case Uint8:
        wordsize = 1;
        break;
    case Int16:
        wordsize = 2;
        m_streamDescription.mFormatFlags |= kAudioFormatFlagIsSignedInteger;
        break;
    case Int24:
        wordsize = 3;
        m_streamDescription.mFormatFlags |= kAudioFormatFlagIsSignedInteger;
        break;
    case Int32:
        wordsize = 4;
        m_streamDescription.mFormatFlags |= kAudioFormatFlagIsSignedInteger;
        break;
    case Float32:
        wordsize = 4;
        m_streamDescription.mFormatFlags |= kAudioFormatFlagIsFloat;
        break;
    case Float64:
        wordsize = 8;
        m_streamDescription.mFormatFlags |= kAudioFormatFlagIsFloat;
        break;
    case None:
        ASSERT_NOT_REACHED();
        wordsize = 0;
        break;
    }

    m_streamDescription.mBitsPerChannel = wordsize * 8;
    if (isInterleaved == IsInterleaved::Yes)
        m_streamDescription.mBytesPerFrame = m_streamDescription.mBytesPerPacket = wordsize * numChannels;
    else {
        m_streamDescription.mFormatFlags |= kAudioFormatFlagIsNonInterleaved;
        m_streamDescription.mBytesPerFrame = m_streamDescription.mBytesPerPacket = wordsize;
    }
}

const PlatformDescription& CAAudioStreamDescription::platformDescription() const
{
    m_platformDescription = { PlatformDescription::CAAudioStreamBasicType, &m_streamDescription };
    return m_platformDescription;
}

AudioStreamDescription::PCMFormat CAAudioStreamDescription::format() const
{
    if (m_format != None)
        return m_format;
    if (m_streamDescription.mFormatID != kAudioFormatLinearPCM)
        return None;
    if (m_streamDescription.mFramesPerPacket != 1)
        return None;
    if (m_streamDescription.mBytesPerFrame != m_streamDescription.mBytesPerPacket)
        return None;
    if (!m_streamDescription.mChannelsPerFrame)
        return None;
    if (!isNativeEndian())
        return None;
    if (m_streamDescription.mBitsPerChannel % 8)
        return None;
    uint32_t bytesPerSample = m_streamDescription.mBitsPerChannel / 8;
    uint32_t numChannelsPerFrame = numberOfInterleavedChannels();
    if (m_streamDescription.mBytesPerFrame % numChannelsPerFrame)
        return None;
    if (m_streamDescription.mBytesPerFrame / numChannelsPerFrame != bytesPerSample)
        return None;
    std::optional<PCMFormat> format;
    auto asbdFormat = m_streamDescription.mFormatFlags & (kLinearPCMFormatFlagIsFloat | kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagsSampleFractionMask);
    if (asbdFormat == kLinearPCMFormatFlagIsFloat) {
        if (bytesPerSample == sizeof(float))
            format = Float32;
        else if (bytesPerSample == sizeof(double))
            format = Float64;
    } else if (asbdFormat == kLinearPCMFormatFlagIsSignedInteger) {
        if (bytesPerSample == sizeof(int16_t))
            format = Int16;
        else if (bytesPerSample == sizeof(int32_t))
            format = Int32;
        else if (bytesPerSample == 3)
            format = Int24;
    } else if (bytesPerSample == sizeof(uint8_t))
        format = Uint8;

    if (!format)
        return None;
    m_format = *format;
    return m_format;
}

bool operator==(const AudioStreamBasicDescription& a, const AudioStreamBasicDescription& b)
{
    return a.mSampleRate == b.mSampleRate
        && a.mFormatID == b.mFormatID
        && a.mFormatFlags == b.mFormatFlags
        && a.mBytesPerPacket == b.mBytesPerPacket
        && a.mFramesPerPacket == b.mFramesPerPacket
        && a.mBytesPerFrame == b.mBytesPerFrame
        && a.mChannelsPerFrame == b.mChannelsPerFrame
        && a.mBitsPerChannel == b.mBitsPerChannel;
}

double CAAudioStreamDescription::sampleRate() const
{
    return m_streamDescription.mSampleRate;
}

bool CAAudioStreamDescription::isPCM() const
{
    return m_streamDescription.mFormatID == kAudioFormatLinearPCM;
}

bool CAAudioStreamDescription::isInterleaved() const
{
    return !(m_streamDescription.mFormatFlags & kAudioFormatFlagIsNonInterleaved);
}

bool CAAudioStreamDescription::isSignedInteger() const
{
    return isPCM() && (m_streamDescription.mFormatFlags & kAudioFormatFlagIsSignedInteger);
}

bool CAAudioStreamDescription::isFloat() const
{
    return isPCM() && (m_streamDescription.mFormatFlags & kAudioFormatFlagIsFloat);
}

bool CAAudioStreamDescription::isNativeEndian() const
{
    return isPCM() && (m_streamDescription.mFormatFlags & kAudioFormatFlagIsBigEndian) == kAudioFormatFlagsNativeEndian;
}

uint32_t CAAudioStreamDescription::numberOfInterleavedChannels() const
{
    return isInterleaved() ? m_streamDescription.mChannelsPerFrame : 1;
}

uint32_t CAAudioStreamDescription::numberOfChannelStreams() const
{
    return isInterleaved() ? 1 : m_streamDescription.mChannelsPerFrame;
}

uint32_t CAAudioStreamDescription::numberOfChannels() const
{
    return m_streamDescription.mChannelsPerFrame;
}

uint32_t CAAudioStreamDescription::sampleWordSize() const
{
    return (m_streamDescription.mBytesPerFrame > 0 && numberOfInterleavedChannels()) ? m_streamDescription.mBytesPerFrame / numberOfInterleavedChannels() :  0;
}

uint32_t CAAudioStreamDescription::bytesPerFrame() const
{
    return m_streamDescription.mBytesPerFrame;
}

uint32_t CAAudioStreamDescription::bytesPerPacket() const
{
    return m_streamDescription.mBytesPerPacket;
}

uint32_t CAAudioStreamDescription::formatFlags() const
{
    return m_streamDescription.mFormatFlags;
}

bool CAAudioStreamDescription::operator==(const AudioStreamBasicDescription& other) const
{
    return m_streamDescription == other;
}

bool CAAudioStreamDescription::operator==(const AudioStreamDescription& other) const
{
    if (other.platformDescription().type != PlatformDescription::CAAudioStreamBasicType)
        return false;

    return operator==(*std::get<const AudioStreamBasicDescription*>(other.platformDescription().description));
}

const AudioStreamBasicDescription& CAAudioStreamDescription::streamDescription() const
{
    return m_streamDescription;
}

AudioStreamBasicDescription& CAAudioStreamDescription::streamDescription()
{
    return m_streamDescription;
}

}

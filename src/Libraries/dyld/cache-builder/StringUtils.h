/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 5, 2024.
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
#ifndef StringUtils_h
#define StringUtils_h

#include <algorithm>
#include <string>

inline bool startsWith(const std::string& str, const std::string& prefix)
{
    return std::mismatch(prefix.begin(), prefix.end(), str.begin()).first == prefix.end();
}

inline bool startsWith(const std::string_view& str, const std::string_view& prefix)
{
    return std::mismatch(prefix.begin(), prefix.end(), str.begin()).first == prefix.end();
}

inline bool startsWith(const char* str, const char* prefix)
{
    return startsWith(std::string_view(str), std::string_view(prefix));
}

inline bool endsWith(const std::string& str, const std::string& suffix)
{
    std::size_t index = str.find(suffix, str.size() - suffix.size());
    return (index != std::string::npos);
}

inline bool endsWith(const std::string_view& str, const std::string_view& suffix)
{
    std::size_t index = str.find(suffix, str.size() - suffix.size());
    return (index != std::string::npos);
}

inline bool contains(const std::string& str, const std::string& search)
{
    std::size_t index = str.find(search);
    return (index != std::string::npos);
}

inline char hexDigit(uint8_t value)
{
    if ( value < 10 )
        return '0' + value;
    else
        return 'a' + value - 10;
}

inline void bytesToHex(const uint8_t* bytes, size_t byteCount, char buffer[])
{
    char* p = buffer;
    for (size_t i=0; i < byteCount; ++i) {
        *p++ = hexDigit(bytes[i] >> 4);
        *p++ = hexDigit(bytes[i] & 0x0F);
    }
    *p++ =  '\0';
}

inline void putHexNibble(uint8_t value, char*& p)
{
    if ( value < 10 )
        *p++ = '0' + value;
    else
        *p++ = 'A' + value - 10;
}

inline void putHexByte(uint8_t value, char*& p)
{
    value &= 0xFF;
    putHexNibble(value >> 4,   p);
    putHexNibble(value & 0x0F, p);
}

inline bool hexCharToUInt(const char hexByte, uint8_t& value) {
    if (hexByte >= '0' && hexByte <= '9') {
        value = hexByte - '0';
        return true;
    } else if (hexByte >= 'A' && hexByte <= 'F') {
        value = hexByte - 'A' + 10;
        return true;
    } else if (hexByte >= 'a' && hexByte <= 'f') {
        value = hexByte - 'a' + 10;
        return true;
    }

    return false;
}

inline uint64_t hexToUInt64(const char* startHexByte, const char** endHexByte) {
    const char* scratch;
    if (endHexByte == nullptr) {
        endHexByte = &scratch;
    }
    if (startHexByte == nullptr)
        return 0;
    uint64_t retval = 0;
    if (startHexByte[0] == '0' &&  startHexByte[1] == 'x') {
        startHexByte +=2;
    }
    *endHexByte = startHexByte + 16;

    //FIXME overrun?
    for (uint32_t i = 0; i < 16; ++i) {
        uint8_t value;
        if (!hexCharToUInt(startHexByte[i], value)) {
            *endHexByte = &startHexByte[i];
            break;
        }
        retval = (retval << 4) + value;
    }
    return retval;
}

inline bool hexStringToBytes(const char* hexString, uint8_t buffer[], unsigned bufferMaxSize, unsigned& bufferLenUsed)
{
    bufferLenUsed = 0;
    bool high = true;
    for (const char* s=hexString; *s != '\0'; ++s) {
        if ( bufferLenUsed > bufferMaxSize )
            return false;
        uint8_t value;
        if ( !hexCharToUInt(*s, value) )
            return false;
        if ( high )
            buffer[bufferLenUsed] = value << 4;
        else
            buffer[bufferLenUsed++] |= value;
        high = !high;
    }
    return true;
}

#endif // StringUtils_h


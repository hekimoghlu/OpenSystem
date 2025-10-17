/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 21, 2023.
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

#if OS(UNIX)
#include <arpa/inet.h>
#endif

namespace WTF {

inline uint32_t wordSwap32(uint32_t x) { return ((x & 0xffff0000) >> 16) | ((x & 0x0000ffff) << 16); }
inline uint64_t byteSwap64(uint64_t x) { return ((x & 0xff00000000000000ULL) >> 56) | ((x & 0x00ff000000000000ULL) >> 40) | ((x & 0x0000ff0000000000ULL) >> 24) | ((x & 0x000000ff00000000ULL) >> 8) | ((x & 0x00000000ff000000ULL) << 8) | ((x & 0x0000000000ff0000ULL) << 24) | ((x & 0x000000000000ff00ULL) << 40) | ((x & 0x00000000000000ffULL) << 56); }
inline uint32_t byteSwap32(uint32_t x) { return ((x & 0xff000000) >> 24) | ((x & 0x00ff0000) >> 8) | ((x & 0x0000ff00) << 8) | ((x & 0x000000ff) << 24); }
inline uint16_t byteSwap16(uint16_t x) { return ((x & 0xff00) >> 8) | ((x & 0x00ff) << 8); }

} // namespace WTF

#if OS(WINDOWS)

#if CPU(BIG_ENDIAN)
inline uint16_t ntohs(uint16_t x) { return x; }
inline uint16_t htons(uint16_t x) { return x; }
inline uint32_t ntohl(uint32_t x) { return x; }
inline uint32_t htonl(uint32_t x) { return x; }
#elif CPU(MIDDLE_ENDIAN)
inline uint16_t ntohs(uint16_t x) { return x; }
inline uint16_t htons(uint16_t x) { return x; }
inline uint32_t ntohl(uint32_t x) { return WTF::wordSwap32(x); }
inline uint32_t htonl(uint32_t x) { return WTF::wordSwap32(x); }
#else
inline uint16_t ntohs(uint16_t x) { return WTF::byteSwap16(x); }
inline uint16_t htons(uint16_t x) { return WTF::byteSwap16(x); }
inline uint32_t ntohl(uint32_t x) { return WTF::byteSwap32(x); }
inline uint32_t htonl(uint32_t x) { return WTF::byteSwap32(x); }
#endif

#endif // OS(WINDOWS)

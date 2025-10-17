/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 8, 2022.
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
#ifndef _SECURITY_AUTH_CRC_H_
#define _SECURITY_AUTH_CRC_H_

#if defined(__cplusplus)
extern "C" {
#endif

extern const uint64_t _crc_table64[256];
extern const uint64_t xorout;
    
AUTH_INLINE uint64_t
crc64_init(void)
{
    return xorout;
}

AUTH_INLINE uint64_t
crc64_final(uint64_t crc)
{
      return crc ^ xorout;
}
    
AUTH_INLINE AUTH_NONNULL_ALL uint64_t
crc64_update(uint64_t crc, const void *buf, uint64_t len)
{
    const unsigned char * ptr = (const unsigned char *) buf;

    while (len-- > 0) {
        crc = _crc_table64[((crc >> 56) ^ *(ptr++)) & 0xff] ^ (crc << 8);
    }
    
    return crc;
}

AUTH_INLINE uint64_t
crc64(const void *buf, uint64_t len)
{
    uint64_t crc = crc64_init();
    
    crc = crc64_update(crc, buf, len);
    
    crc = crc64_final(crc);
    
    return crc;
}
    
#if defined(__cplusplus)
}
#endif

#endif /* !_SECURITY_AUTH_CRC_H_ */

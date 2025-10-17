/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 19, 2023.
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
#include "curl_setup.h"

#include "curl_endian.h"

/*
 * Curl_read16_le()
 *
 * This function converts a 16-bit integer from the little endian format, as
 * used in the incoming package to whatever endian format we're using
 * natively.
 *
 * Parameters:
 *
 * buf      [in]     - A pointer to a 2 byte buffer.
 *
 * Returns the integer.
 */
unsigned short Curl_read16_le(const unsigned char *buf)
{
  return (unsigned short)(((unsigned short)buf[0]) |
                          ((unsigned short)buf[1] << 8));
}

/*
 * Curl_read32_le()
 *
 * This function converts a 32-bit integer from the little endian format, as
 * used in the incoming package to whatever endian format we're using
 * natively.
 *
 * Parameters:
 *
 * buf      [in]     - A pointer to a 4 byte buffer.
 *
 * Returns the integer.
 */
unsigned int Curl_read32_le(const unsigned char *buf)
{
  return ((unsigned int)buf[0]) | ((unsigned int)buf[1] << 8) |
         ((unsigned int)buf[2] << 16) | ((unsigned int)buf[3] << 24);
}

/*
 * Curl_read16_be()
 *
 * This function converts a 16-bit integer from the big endian format, as
 * used in the incoming package to whatever endian format we're using
 * natively.
 *
 * Parameters:
 *
 * buf      [in]     - A pointer to a 2 byte buffer.
 *
 * Returns the integer.
 */
unsigned short Curl_read16_be(const unsigned char *buf)
{
  return (unsigned short)(((unsigned short)buf[0] << 8) |
                          ((unsigned short)buf[1]));
}

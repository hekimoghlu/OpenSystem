/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 24, 2025.
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
#ifdef HAVE_CONFIG_H
#include "config.h"
RCSID("$Id$");
#endif

#include "otp_locl.h"

extern const char *const std_dict[];

unsigned
otp_checksum (OtpKey key)
{
  int i;
  unsigned sum = 0;

  for (i = 0; i < OTPKEYSIZE; ++i)
    sum += ((key[i] >> 0) & 0x03)
      + ((key[i] >> 2) & 0x03)
      + ((key[i] >> 4) & 0x03)
      + ((key[i] >> 6) & 0x03);
  sum &= 0x03;
  return sum;
}

void
otp_print_stddict (OtpKey key, char *str, size_t sz)
{
  unsigned sum;

  sum = otp_checksum (key);
  snprintf (str, sz,
	    "%s %s %s %s %s %s",
	    std_dict[(key[0] << 3) | (key[1] >> 5)],
	    std_dict[((key[1] & 0x1F) << 6) | (key[2] >> 2)],
	    std_dict[((key[2] & 0x03) << 9) | (key[3] << 1) | (key[4] >> 7)],
	    std_dict[((key[4] & 0x7F) << 4) | (key[5] >> 4)],
	    std_dict[((key[5] & 0x0F) << 7) | (key[6] >> 1)],
	    std_dict[((key[6] & 0x01) << 10) | (key[7] << 2) | sum]);
}

void
otp_print_hex (OtpKey key, char *str, size_t sz)
{
  snprintf (str, sz,
	    "%02x%02x%02x%02x%02x%02x%02x%02x",
	    key[0], key[1], key[2], key[3],
	    key[4], key[5], key[6], key[7]);
}

void
otp_print_hex_extended (OtpKey key, char *str, size_t sz)
{
  strlcpy (str, OTP_HEXPREFIX, sz);
  otp_print_hex (key,
		 str + strlen(OTP_HEXPREFIX),
		 sz - strlen(OTP_HEXPREFIX));
}

void
otp_print_stddict_extended (OtpKey key, char *str, size_t sz)
{
  strlcpy (str, OTP_WORDPREFIX, sz);
  otp_print_stddict (key,
		     str + strlen(OTP_WORDPREFIX),
		     sz - strlen(OTP_WORDPREFIX));
}

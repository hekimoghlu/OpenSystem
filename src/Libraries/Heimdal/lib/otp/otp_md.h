/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 29, 2023.
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
/* $Id$ */

int otp_md4_init (OtpKey key, const char *pwd, const char *seed);
int otp_md4_hash (const char *, size_t, unsigned char *res);
int otp_md4_next (OtpKey key);

int otp_md5_init (OtpKey key, const char *pwd, const char *seed);
int otp_md5_hash (const char *, size_t, unsigned char *res);
int otp_md5_next (OtpKey key);

int otp_sha_init (OtpKey key, const char *pwd, const char *seed);
int otp_sha_hash (const char *, size_t, unsigned char *res);
int otp_sha_next (OtpKey key);

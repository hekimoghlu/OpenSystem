/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 28, 2021.
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

#ifndef _OTP_H
#define _OTP_H

#include <stdlib.h>
#include <time.h>

enum {OTPKEYSIZE = 8};

typedef unsigned char OtpKey[OTPKEYSIZE];

#define OTP_MIN_PASSPHRASE 10
#define OTP_MAX_PASSPHRASE 63

#define OTP_USER_TIMEOUT   120
#define OTP_DB_TIMEOUT      60

#define OTP_HEXPREFIX "hex:"
#define OTP_WORDPREFIX "word:"

typedef enum { OTP_ALG_MD4, OTP_ALG_MD5, OTP_ALG_SHA } OtpAlgID;

#define OTP_ALG_DEFAULT "md5"

typedef struct {
  OtpAlgID id;
  char *name;
  int hashsize;
  int (*hash)(const char *, size_t, unsigned char *);
  int (*init)(OtpKey, const char *, const char *);
  int (*next)(OtpKey);
} OtpAlgorithm;

typedef struct {
  char *user;
  OtpAlgorithm *alg;
  unsigned n;
  char seed[17];
  OtpKey key;
  int challengep;
  time_t lock_time;
  char *err;
} OtpContext;

OtpAlgorithm *otp_find_alg (char *);
void otp_print_stddict (OtpKey, char *, size_t);
void otp_print_hex (OtpKey, char *, size_t);
void otp_print_stddict_extended (OtpKey, char *, size_t);
void otp_print_hex_extended (OtpKey, char *, size_t);
unsigned otp_checksum (OtpKey);
int otp_parse_hex (OtpKey, const char *);
int otp_parse_stddict (OtpKey, const char *);
int otp_parse_altdict (OtpKey, const char *, OtpAlgorithm *);
int otp_parse (OtpKey, const char *, OtpAlgorithm *);
int otp_challenge (OtpContext *, char *, char *, size_t);
int otp_verify_user (OtpContext *, const char *);
int otp_verify_user_1 (OtpContext *, const char *);
char *otp_error (OtpContext *);

void *otp_db_open (void);
void otp_db_close (void *);
int otp_put (void *, OtpContext *);
int otp_get (void *, OtpContext *);
int otp_simple_get (void *, OtpContext *);
int otp_delete (void *, OtpContext *);

#endif /* _OTP_H */

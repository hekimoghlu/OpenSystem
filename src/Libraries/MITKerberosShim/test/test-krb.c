/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 26, 2022.
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
#include <stdio.h>
#include <stdlib.h>

#include <CoreFoundation/CoreFoundation.h>
#include <krb5.h>

#include "test.h"

/* From Krb4DeprecatedAPIs.c */
struct ktext {
    int     length;
    unsigned char dat[MAX_KTXT_LEN];
    unsigned long mbz;
};

int krb_get_cred(char*, char*, char*, void*);
const char * krb_get_err_text(int);
int krb_get_lrealm (char*, int);
int krb_get_tf_fullname (const char*,char*, char*, char*);
int krb_mk_req(struct ktext, char*, char*, char*, UInt32);
char *krb_realmofhost(char*);
const char *tkt_string (void);
void com_err(const char *progname, errcode_t code, const char *format, ...);

typedef unsigned char des_cblock[8];
typedef des_cblock mit_des_cblock;
#define DES_INT32 int
unsigned long des_quad_cksum(const unsigned char *in, unsigned DES_INT32 *out, long length, int out_count, mit_des_cblock *c_seed);

int main(int argc, char **argv)
{
	struct ktext ss;
    memset(&ss, 0, sizeof(ss));

	VERIFY_DEPRECATED_I(
		"krb_get_cred",
		krb_get_cred(NULL,NULL,NULL,NULL));

	VERIFY_DEPRECATED_S(
		"krb_get_err_text",
		krb_get_err_text(1));

	VERIFY_DEPRECATED_I(
		"krb_get_lrealm",
		krb_get_lrealm(NULL, 1));

	VERIFY_DEPRECATED_I(
		"krb_get_tf_fullname",
		krb_get_tf_fullname(NULL,NULL,NULL,NULL));

	VERIFY_DEPRECATED_I(
		"krb_mk_req",
		krb_mk_req(ss, NULL, NULL, NULL, 0));

	VERIFY_DEPRECATED_S(
		"krb_realmofhost",
		krb_realmofhost(NULL));

	com_err("program", 0, "format");

	/*
	VERIFY_DEPRECATED_S(
		"des_quad_cksum",
		des_quad_chsum(NULL, 0, 0, 0, NULL));
	*/
	VERIFY_DEPRECATED_S(
		"tkt_string",
		tkt_string());

	printf("Test completed.\n");
	return 0;
}

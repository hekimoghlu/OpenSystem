/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 17, 2023.
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
/* $OpenBSD: krl.h,v 1.5 2015/12/30 23:46:14 djm Exp $ */

#ifndef _KRL_H
#define _KRL_H

/* Functions to manage key revocation lists */

#define KRL_MAGIC		"SSHKRL\n\0"
#define KRL_FORMAT_VERSION	1

/* KRL section types */
#define KRL_SECTION_CERTIFICATES	1
#define KRL_SECTION_EXPLICIT_KEY	2
#define KRL_SECTION_FINGERPRINT_SHA1	3
#define KRL_SECTION_SIGNATURE		4

/* KRL_SECTION_CERTIFICATES subsection types */
#define KRL_SECTION_CERT_SERIAL_LIST	0x20
#define KRL_SECTION_CERT_SERIAL_RANGE	0x21
#define KRL_SECTION_CERT_SERIAL_BITMAP	0x22
#define KRL_SECTION_CERT_KEY_ID		0x23

struct sshkey;
struct sshbuf;
struct ssh_krl;

struct ssh_krl *ssh_krl_init(void);
void ssh_krl_free(struct ssh_krl *krl);
void ssh_krl_set_version(struct ssh_krl *krl, u_int64_t version);
int ssh_krl_set_comment(struct ssh_krl *krl, const char *comment);
int ssh_krl_revoke_cert_by_serial(struct ssh_krl *krl,
    const struct sshkey *ca_key, u_int64_t serial);
int ssh_krl_revoke_cert_by_serial_range(struct ssh_krl *krl,
    const struct sshkey *ca_key, u_int64_t lo, u_int64_t hi);
int ssh_krl_revoke_cert_by_key_id(struct ssh_krl *krl,
    const struct sshkey *ca_key, const char *key_id);
int ssh_krl_revoke_key_explicit(struct ssh_krl *krl, const struct sshkey *key);
int ssh_krl_revoke_key_sha1(struct ssh_krl *krl, const struct sshkey *key);
int ssh_krl_revoke_key(struct ssh_krl *krl, const struct sshkey *key);
int ssh_krl_to_blob(struct ssh_krl *krl, struct sshbuf *buf,
    const struct sshkey **sign_keys, u_int nsign_keys);
int ssh_krl_from_blob(struct sshbuf *buf, struct ssh_krl **krlp,
    const struct sshkey **sign_ca_keys, size_t nsign_ca_keys);
int ssh_krl_check_key(struct ssh_krl *krl, const struct sshkey *key);
int ssh_krl_file_contains_key(const char *path, const struct sshkey *key);

#endif /* _KRL_H */


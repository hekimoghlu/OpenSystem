/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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
#include <sys/types.h>
#include <GSS/gssapi.h>
#include <GSS/gssapi_krb5.h>
#include <GSS/gssapi_ntlm.h>
#include <GSS/gssapi_spnego.h>
#include <Kernel/gssd/gssd_mach_types.h>
#include <sys/queue.h>

struct smb_gss_cred_list_entry {
	TAILQ_ENTRY(smb_gss_cred_list_entry) next;
	char *principal;
	uint32_t expire;
	gss_OID mech;
};

TAILQ_HEAD(smb_gss_cred_list, smb_gss_cred_list_entry);

struct smb_gss_cred_ctx {
	dispatch_semaphore_t sem;
	uint32_t maj;
	gss_cred_id_t creds;
};

int smb_gss_get_cred_list(struct smb_gss_cred_list **, const gss_OID /*mech*/);
void smb_gss_free_cred_entry(struct smb_gss_cred_list_entry **);
void smb_gss_free_cred_list(struct smb_gss_cred_list **);
int smb_gss_match_cred_entry(struct smb_gss_cred_list_entry *, const gss_OID /*mech*/,
			const char * /*name*/, const char * /*domain*/);

char *smb_gss_principal_from_cred(void *);
void smb_release_gss_cred(void *, int);
int smb_acquire_ntlm_cred(const char *, const char *, const char *, void **);
int smb_acquire_krb5_cred(const char *, const char *, const char *, void **);
CFStringRef TargetNameCreateWithHostName(struct smb_ctx *ctx);
void GetTargetNameUsingHostName(struct smb_ctx *);
int serverSupportsKerberos(CFDictionaryRef);

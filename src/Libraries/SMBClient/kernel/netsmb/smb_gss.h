/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 16, 2023.
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
#ifndef SMB_GSS_H
#define SMB_GSS_H

#define GSS_MACH_MAX_RETRIES 3
#define SKEYLEN 8

#ifndef GSS_S_COMPLETE
#define GSS_S_COMPLETE 0
#endif

#ifndef GSS_S_CONTINUE_NEEDED
#define GSS_S_CONTINUE_NEEDED 1
#endif

#define SMB_USE_GSS(vp) (IPC_PORT_VALID((vp)->gss_mp))
#define SMB_GSS_CONTINUE_NEEDED(p) ((p)->gss_major == GSS_S_CONTINUE_NEEDED)
#define SMB_GSS_ERROR(p) ((p)->gss_major != GSS_S_COMPLETE && \
	(p)->gss_major != GSS_S_CONTINUE_NEEDED)
#define SMB_GSS_COMPLETE(p) ((p)->gss_major == GSS_S_COMPLETE)
int  smb_gss_alt_ch_session_setup(struct smbiod *iod);
int  smb_gss_negotiate(struct smbiod *iod, vfs_context_t context);
int  smb_gss_ssnsetup(struct smbiod *iod, vfs_context_t context);
void smb_gss_destroy(struct smb_gss *gp);

void smb_gss_ref_cred(struct smbiod *iod);
void smb_gss_rel_cred(struct smbiod *iod);
int  smb_gss_dup(struct smb_gss *parent_gp, struct smb_gss *new_gp);

#ifdef SMB_DEBUG
//#define DEBUG_TURN_OFF_EXT_SEC 1
#endif // SMB_DEBUG

#endif

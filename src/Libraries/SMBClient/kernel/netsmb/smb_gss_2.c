/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 13, 2021.
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
#include <sys/smb_apple.h>

#include <netsmb/smb.h>
#include <netsmb/smb_2.h>
#include <netsmb/smb_rq.h>
#include <netsmb/smb_rq_2.h>
#include <netsmb/smb_conn.h>
#include <netsmb/smb_conn_2.h>
#include <netsmb/smb_gss_2.h>

int
smb_gss_ssandx(struct smbiod *iod, uint32_t caps, uint16_t *action,
               vfs_context_t context)
{
    int retval;
    uint16_t sess_flags = 0;
    struct smb_session *sessionp = iod->iod_session; 
    
    if (sessionp->session_flags & SMBV_SMB2) {
        retval = smb2_smb_gss_session_setup(iod, &sess_flags, context);
        if (retval == 0) {
            if (iod->iod_flags & SMBIOD_ALTERNATE_CHANNEL) {
                if (sessionp->session_sopt.sv_sessflags & SMB2_SESSION_FLAG_IS_GUEST) {
                    SMBERROR("id %u SMB2_SESSION_FLAG_IS_GUEST should not be set on alt ch (0x%x)",
                             iod->iod_id, sessionp->session_sopt.sv_sessflags);
                    return EINVAL;
                }

                if ((sessionp->session_sopt.sv_sessflags & SMB2_SESSION_FLAG_IS_NULL) != (sess_flags & SMB2_SESSION_FLAG_IS_NULL)) {
                    // Just warnout no action needed
                    SMBWARNING("id %u ALternate channel SMB2_SESSION_FLAG_IS_NULL mismatch (0x%x, 0x%x)",
                             iod->iod_id, sessionp->session_sopt.sv_sessflags, sess_flags);
                }

                /*
                 * Ignore SMB2_SESSION_FLAG_ENCRYPT_DATA mismatch, since win server 2019 does not set
                 * this bit on alternate channel.
                 */

            } else {
                /* Save Flags field from Session Setup reply */
                sessionp->session_sopt.sv_sessflags = sess_flags;
            }

            /* Remap SMB 2/3 session flags to SMB 1 action flags */
            if (sess_flags & SMB2_SESSION_FLAG_IS_GUEST) {
                /* Return that we got logged in as Guest */
                *action |= SMB_ACT_GUEST;
            }
        }
    }
    else {
        retval = smb1_gss_ssandx(sessionp, caps, action, context);
    }
    return (retval);
}


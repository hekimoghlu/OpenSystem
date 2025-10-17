/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 26, 2022.
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
#include <unistd.h>
#include <stdarg.h>
#include <sys/msg.h>

/*
 * Stub function to account for the differences in the ipc_perm structure,
 * while maintaining binary backward compatibility.
 *
 * This is only the legacy behavior.
 */
extern int __msgctl(int, int, struct msqid_ds *);

int
msgctl(int msqid, int cmd, struct msqid_ds *ds)
{
	struct __msqid_ds_old	*ds_old = (struct __msqid_ds_old *)ds;
	struct __msqid_ds_new	ds2;
	struct __msqid_ds_new	*ds_new = &ds2;
	int			rv;

#define	_UP_CVT(x)	ds_new-> x = ds_old-> x
#define	_DN_CVT(x)	ds_old-> x = ds_new-> x

	if (cmd == IPC_SET) {
		/* convert before call */
		_UP_CVT(msg_perm.uid);
		_UP_CVT(msg_perm.gid);
		_UP_CVT(msg_perm.cuid);
		_UP_CVT(msg_perm.cgid);
		_UP_CVT(msg_perm.mode);
		ds_new->msg_perm._seq = ds_old->msg_perm.seq;
		ds_new->msg_perm._key = ds_old->msg_perm.key;
		_UP_CVT(msg_first);
		_UP_CVT(msg_last);
		_UP_CVT(msg_cbytes);
		_UP_CVT(msg_qnum);
		_UP_CVT(msg_qbytes);
		_UP_CVT(msg_lspid);
		_UP_CVT(msg_lrpid);
		_UP_CVT(msg_stime);
		_UP_CVT(msg_pad1);	/* binary compatibility */
		_UP_CVT(msg_rtime);
		_UP_CVT(msg_pad2);	/* binary compatibility */
		_UP_CVT(msg_ctime);
		_UP_CVT(msg_pad3);	/* binary compatibility */
		_UP_CVT(msg_pad4[0]);	/* binary compatibility */
		_UP_CVT(msg_pad4[1]);	/* binary compatibility */
		_UP_CVT(msg_pad4[2]);	/* binary compatibility */
		_UP_CVT(msg_pad4[3]);	/* binary compatibility */
	}

	rv = __msgctl(msqid, cmd, ds_new);

	if (cmd == IPC_STAT) {
		/* convert after call */
		_DN_CVT(msg_perm.uid);	/* warning!  precision loss! */
		_DN_CVT(msg_perm.gid);	/* warning!  precision loss! */
		_DN_CVT(msg_perm.cuid);	/* warning!  precision loss! */
		_DN_CVT(msg_perm.cgid);	/* warning!  precision loss! */
		_DN_CVT(msg_perm.mode);
		ds_old->msg_perm.seq = ds_new->msg_perm._seq;
		ds_old->msg_perm.key = ds_new->msg_perm._key;
		_DN_CVT(msg_first);
		_DN_CVT(msg_last);
		_DN_CVT(msg_cbytes);
		_DN_CVT(msg_qnum);
		_DN_CVT(msg_qbytes);
		_DN_CVT(msg_lspid);
		_DN_CVT(msg_lrpid);
		_DN_CVT(msg_stime);
		_DN_CVT(msg_pad1);	/* binary compatibility */
		_DN_CVT(msg_rtime);
		_DN_CVT(msg_pad2);	/* binary compatibility */
		_DN_CVT(msg_ctime);
		_DN_CVT(msg_pad3);	/* binary compatibility */
		_DN_CVT(msg_pad4[0]);	/* binary compatibility */
		_DN_CVT(msg_pad4[1]);	/* binary compatibility */
		_DN_CVT(msg_pad4[2]);	/* binary compatibility */
		_DN_CVT(msg_pad4[3]);	/* binary compatibility */
	}

	return (rv);
}

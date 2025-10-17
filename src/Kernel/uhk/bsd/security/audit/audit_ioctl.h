/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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
#ifndef _SECURITY_AUDIT_AUDIT_IOCTL_H_
#define _SECURITY_AUDIT_AUDIT_IOCTL_H_

#include <bsm/audit.h>

#define AUDITPIPE_IOBASE        'A'
#define AUDITSDEV_IOBASE        'S'

/*
 * Data structures used for complex ioctl arguments.  Do not change existing
 * structures, add new revised ones to be used by new ioctls, and keep the
 * old structures and ioctls for backwards compatibility.
 */
struct auditpipe_ioctl_preselect {
	au_id_t         aip_auid;
	au_mask_t       aip_mask;
};

/*
 * Possible modes of operation for audit pipe preselection.
 */
#define AUDITPIPE_PRESELECT_MODE_TRAIL  1       /* Global audit trail. */
#define AUDITPIPE_PRESELECT_MODE_LOCAL  2       /* Local audit trail. */

/*
 * Ioctls to read and control the behavior of individual audit pipe devices.
 */
#define AUDITPIPE_GET_QLEN              _IOR(AUDITPIPE_IOBASE, 1, u_int)
#define AUDITPIPE_GET_QLIMIT            _IOR(AUDITPIPE_IOBASE, 2, u_int)
#define AUDITPIPE_SET_QLIMIT            _IOW(AUDITPIPE_IOBASE, 3, u_int)
#define AUDITPIPE_GET_QLIMIT_MIN        _IOR(AUDITPIPE_IOBASE, 4, u_int)
#define AUDITPIPE_GET_QLIMIT_MAX        _IOR(AUDITPIPE_IOBASE, 5, u_int)
#define AUDITPIPE_GET_PRESELECT_FLAGS   _IOR(AUDITPIPE_IOBASE, 6, au_mask_t)
#define AUDITPIPE_SET_PRESELECT_FLAGS   _IOW(AUDITPIPE_IOBASE, 7, au_mask_t)
#define AUDITPIPE_GET_PRESELECT_NAFLAGS _IOR(AUDITPIPE_IOBASE, 8, au_mask_t)
#define AUDITPIPE_SET_PRESELECT_NAFLAGS _IOW(AUDITPIPE_IOBASE, 9, au_mask_t)
#define AUDITPIPE_GET_PRESELECT_AUID    _IOR(AUDITPIPE_IOBASE, 10,      \
	                                    struct auditpipe_ioctl_preselect)
#define AUDITPIPE_SET_PRESELECT_AUID    _IOW(AUDITPIPE_IOBASE, 11,      \
	                                    struct auditpipe_ioctl_preselect)
#define AUDITPIPE_DELETE_PRESELECT_AUID _IOW(AUDITPIPE_IOBASE, 12, au_id_t)
#define AUDITPIPE_FLUSH_PRESELECT_AUID  _IO(AUDITPIPE_IOBASE, 13)
#define AUDITPIPE_GET_PRESELECT_MODE    _IOR(AUDITPIPE_IOBASE, 14, int)
#define AUDITPIPE_SET_PRESELECT_MODE    _IOW(AUDITPIPE_IOBASE, 15, int)
#define AUDITPIPE_FLUSH                 _IO(AUDITPIPE_IOBASE, 16)
#define AUDITPIPE_GET_MAXAUDITDATA      _IOR(AUDITPIPE_IOBASE, 17, u_int)

/*
 * Ioctls to retrieve audit pipe statistics.
 */
#define AUDITPIPE_GET_INSERTS           _IOR(AUDITPIPE_IOBASE, 100, u_int64_t)
#define AUDITPIPE_GET_READS             _IOR(AUDITPIPE_IOBASE, 101, u_int64_t)
#define AUDITPIPE_GET_DROPS             _IOR(AUDITPIPE_IOBASE, 102, u_int64_t)
#define AUDITPIPE_GET_TRUNCATES         _IOR(AUDITPIPE_IOBASE, 103, u_int64_t)

/*
 * Ioctls for the audit session device.
 */
#define AUDITSDEV_GET_QLEN              _IOR(AUDITSDEV_IOBASE, 1, u_int)
#define AUDITSDEV_GET_QLIMIT            _IOR(AUDITSDEV_IOBASE, 2, u_int)
#define AUDITSDEV_SET_QLIMIT            _IOW(AUDITSDEV_IOBASE, 3, u_int)
#define AUDITSDEV_GET_QLIMIT_MIN        _IOR(AUDITSDEV_IOBASE, 4, u_int)
#define AUDITSDEV_GET_QLIMIT_MAX        _IOR(AUDITSDEV_IOBASE, 5, u_int)
#define AUDITSDEV_FLUSH                 _IO(AUDITSDEV_IOBASE, 6)
#define AUDITSDEV_GET_MAXDATA           _IOR(AUDITSDEV_IOBASE, 7, u_int)

/*
 * Ioctls to retrieve and set the ALLSESSIONS flag in the audit session device.
 */
#define AUDITSDEV_GET_ALLSESSIONS       _IOR(AUDITSDEV_IOBASE, 100, u_int)
#define AUDITSDEV_SET_ALLSESSIONS       _IOW(AUDITSDEV_IOBASE, 101, u_int)

/*
 * Ioctls to retrieve audit sessions device statistics.
 */
#define AUDITSDEV_GET_INSERTS           _IOR(AUDITSDEV_IOBASE, 200, u_int64_t)
#define AUDITSDEV_GET_READS             _IOR(AUDITSDEV_IOBASE, 201, u_int64_t)
#define AUDITSDEV_GET_DROPS             _IOR(AUDITSDEV_IOBASE, 202, u_int64_t)

#endif /* _SECURITY_AUDIT_AUDIT_IOCTL_H_ */

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 27, 2022.
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
/*
 *  enums.h
 *  bless
 *
 *  Created by Shantonu Sen <ssen@apple.com> on Wed Nov 14 2001.
 *  Copyright (c) 2001-2007 Apple Inc. All Rights Reserved.
 *
 *  $Id: enums.h,v 1.37 2006/07/17 22:19:05 ssen Exp $
 *
 *
 */

#ifndef _ENUMS_H_
#define _ENUMS_H_

enum {
    kdummy = 0,
	kallowui,
	kalternateos,
    kapfsdriver,
	kbootefi,
	kbooter,
    kbootinfo,
	kcreatesnapshot,
	kdevice,
    kfile,
	kfirmware,				/* 10 */
    kfolder,
    kfolder9,
    kgetboot,
    khelp,
    kinfo,
    kkernel,
    kkernelcache,
    klabel,
    klabelfile,
	klastsealedsnapshot,	/* 20 */
    klegacy,
    klegacydrivehint,
    kmkext,
    kmount,
    knetboot,
    knextonly,
    kopenfolder,
    koptions,
    kpasspromt,
    kpayload,				/* 30 */
	kpersonalize,
    kplist,
    kquiet,
	krecovery,
    kreset,
    ksave9,
    ksaveX,
    kserver,
    ksetboot,
    kshortform,				/* 40 */
    kstartupfile,
    kstdinpass,
    kunbless,
    kuser,
    kuse9,
    kusetdmasexternal,
    kverbose,
    kversion,
    ksnapshot,
    ksnapshotname,		/* 50 */
    knoapfsdriver,
    klast
};

// getopt_long(3) uses ':' as a special return value
extern int too_many_options[klast >= ':' ? -1 : 0];

#endif // _ENUMS_H_

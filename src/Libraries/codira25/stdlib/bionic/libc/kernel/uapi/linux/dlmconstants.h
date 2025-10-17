/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 5, 2022.
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
#ifndef __DLMCONSTANTS_DOT_H__
#define __DLMCONSTANTS_DOT_H__
#define DLM_LOCKSPACE_LEN 64
#define DLM_RESNAME_MAXLEN 64
#define DLM_LOCK_IV (- 1)
#define DLM_LOCK_NL 0
#define DLM_LOCK_CR 1
#define DLM_LOCK_CW 2
#define DLM_LOCK_PR 3
#define DLM_LOCK_PW 4
#define DLM_LOCK_EX 5
#define DLM_LKF_NOQUEUE 0x00000001
#define DLM_LKF_CANCEL 0x00000002
#define DLM_LKF_CONVERT 0x00000004
#define DLM_LKF_VALBLK 0x00000008
#define DLM_LKF_QUECVT 0x00000010
#define DLM_LKF_IVVALBLK 0x00000020
#define DLM_LKF_CONVDEADLK 0x00000040
#define DLM_LKF_PERSISTENT 0x00000080
#define DLM_LKF_NODLCKWT 0x00000100
#define DLM_LKF_NODLCKBLK 0x00000200
#define DLM_LKF_EXPEDITE 0x00000400
#define DLM_LKF_NOQUEUEBAST 0x00000800
#define DLM_LKF_HEADQUE 0x00001000
#define DLM_LKF_NOORDER 0x00002000
#define DLM_LKF_ORPHAN 0x00004000
#define DLM_LKF_ALTPR 0x00008000
#define DLM_LKF_ALTCW 0x00010000
#define DLM_LKF_FORCEUNLOCK 0x00020000
#define DLM_LKF_TIMEOUT 0x00040000
#define DLM_ECANCEL 0x10001
#define DLM_EUNLOCK 0x10002
#endif

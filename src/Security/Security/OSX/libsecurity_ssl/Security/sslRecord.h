/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 21, 2025.
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
 * sslRecord.h - SSL Record Layer
 */

#ifndef _SSLRECORD_H_
#define _SSLRECORD_H_ 1

#include "sslPriv.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 
 * Slightly smaller that 16384 to make room for a MAC in an SSL 2.0 
 * 3-byte header record 
 */
#define MAX_RECORD_LENGTH   16300

OSStatus SSLWriteRecord(
    SSLRecord 	rec,
    SSLContext 	*ctx);

OSStatus SSLFreeRecord(
    SSLRecord 	rec,
    SSLContext 	*ctx);

OSStatus SSLReadRecord(
	SSLRecord 	*rec,
	SSLContext 	*ctx);

OSStatus SSLServiceWriteQueue(
    SSLContext  *ctx);

#ifdef __cplusplus
}
#endif

#endif /* _SSLRECORD_H_ */

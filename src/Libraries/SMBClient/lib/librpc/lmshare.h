/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 27, 2023.
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
#ifndef LMSHARE_H_CA194909_8873_4FED_BECC_C9339E215B2D
#define LMSHARE_H_CA194909_8873_4FED_BECC_C9339E215B2D

#ifdef RPC_LIB_ONLY
#include "srvsvc_client.h"
#else // RPC_LIB_ONLY
#include <nt/srvsvc.h>
#endif // RPC_LIB_ONLY

#ifdef __cplusplus
extern "C" {
#endif
	
/* 
 * XXX - Should this be in its own include file?
 * We need to traslate routine to convert these to errno
 */
typedef enum werror
{
#define declare_werror(name, value) name = value,
#include "werror.inc"
} werror;
	
/*
 * MS-SRVS 2.2.4.28 SHARE_INFO_1005 shi1005_flags
 * MS-SMB2 2.2.10 ShareFlags
 */
#define CSC_CACHE_MANUAL_REINIT 0x00000000 // SMB2_SHAREFLAG_MANUAL_CACHING
#define CSC_CACHE_AUTO_REINIT   0x00000010 // SMB2_SHAREFLAG_AUTO_CACHING
#define CSC_CACHE_VDO           0x00000020 // SMB2_SHAREFLAG_VDO_CACHING
#define CSC_CACHE_NONE          0x00000030 // SMB2_SHAREFLAG_NO_CACHING

#define CSC_MASK 0x30

/* Extract the CSC flags from a SH1005 flag set. */
#define SH1005_TO_CSC(flags) ((flags) & CSC_MASK)

NET_API_STATUS
NetShareGetInfo(
				const char * ServerName,
				const char * NetName,
				uint32_t Level,
				PSHARE_INFO * ShareInfo
				);

NET_API_STATUS
NetShareEnum(
			 const char * ServerName,
			 uint32_t Level,
			 PSHARE_ENUM_STRUCT * InfoStruct
			 );

void NetApiBufferFree(void * bufptr);

#ifdef __cplusplus
} // extern "C"
#endif


#endif /* LMSHARE_H_CA194909_8873_4FED_BECC_C9339E215B2D */
/* vim: set sw=4 ts=4 tw=79 et: */

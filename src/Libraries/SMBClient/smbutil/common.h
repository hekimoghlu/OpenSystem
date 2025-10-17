/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 20, 2025.
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
#ifndef __COMMON_H__
#define __COMMON_H__

#include <CoreFoundation/CFDictionary.h>
#include <CoreFoundation/CFArray.h>

#ifdef __cplusplus
extern "C" {
#endif
	
#define iprintf(ident,args...)	do { printf("%-" # ident "s", ""); \
				printf(args);}while(0)

extern int verbose;

enum OutputFormat { None = 0, Json = 1 };

int  cmd_lookup(int argc, char *argv[]);
int  cmd_status(int argc, char *argv[]);
int  cmd_view(int argc, char *argv[]);
int  cmd_dfs(int argc, char *argv[]);
int  cmd_identity(int argc, char *argv[]);
int  cmd_statshares(int argc, char *argv[]);
int  cmd_multichannel(int argc, char *argv[]);
int  cmd_snapshot(int argc, char *argv[]);
int  cmd_smbstat(int argc, char *argv[]);
void lookup_usage(void);
void status_usage(void);
void view_usage(void);
void dfs_usage(void);
void identity_usage(void);
void ntstatus_to_err(NTSTATUS status);
void statshares_usage(void);
void multichannel_usage(void);
void snapshot_usage(void);
void smbstat_usage(void);
struct statfs *smb_getfsstat(int *fs_cnt);
CFArrayRef createShareArrayFromShareDictionary(CFDictionaryRef shareDict);
char * get_share_name(const char *name);
/*
 * Allocate a buffer and then use CFStringGetCString to copy the c-style string
 * into the buffer. The calling routine needs to free the buffer when done.
 */
char *CStringCreateWithCFString(CFStringRef inStr);
	
#ifdef __cplusplus
} // extern "C"
#endif

#endif // __COMMON_H__

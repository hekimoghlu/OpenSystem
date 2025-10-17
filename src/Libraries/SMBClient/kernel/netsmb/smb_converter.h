/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 3, 2023.
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
#ifndef _SMB_CONVERTER_H_
#define _SMB_CONVERTER_H_

/*
 * Don't use the SFM conversion tables, really defined so we know that we 
 * specifically do not want to use the SFM conversions.
 */
#define NO_SFM_CONVERSIONS			0x0000
/*
 * Used when calling the smb_convert_path_to_network and smb_convert_network_to_path
 * routines. Since UTF_SFM_CONVERSIONS is only defined in the kernel. The kernel
 * has it defined as "Use SFM mappings for illegal NTFS chars"
 */
#define SMB_UTF_SFM_CONVERSIONS		0x0020
/*
 * Used when calling the smb_convert_path_to_network and smb_convert_network_to_path
 * routines. Make sure the returned path is a full path, add a starting delimiter
 * if one does exist. The calling process must make sure the buffer is large
 * enough to hold the output plus the terminator size.
 */
#define SMB_FULLPATH_CONVERSIONS	0x0100

#ifdef KERNEL
#include <sys/utfconv.h>


int smb_convert_to_network(const char **inbuf, size_t *inbytesleft, char **outbuf, 
						   size_t *outbytesleft, int flags, int usingUnicode);
int smb_convert_from_network(const char **inbuf, size_t *inbytesleft, char **outbuf, 
							 size_t *outbytesleft, int flags, int usingUnicode);
size_t smb_strtouni(uint16_t *dst, const char *src, size_t inlen, int flags);
size_t smb_unitostr(char *dst, const uint16_t *src, size_t inlen, size_t maxlen, int flags);
size_t smb_utf16_strnsize(const uint16_t *s, size_t n_bytes);
int smb_convert_path_to_network(char *path, size_t max_path_len, char *network, 
								size_t *ntwrk_len, char ntwrk_delimiter, int inflags, 
								int usingUnicode);
int smb_convert_network_to_path(char *network, size_t max_ntwrk_len, char *path, 
							size_t *path_len, char ntwrk_delimiter, int flags, 
							int usingUnicode);
#endif // KERNEL

#endif // _SMB_CONVERTER_H_

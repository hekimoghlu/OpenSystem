/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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
#ifndef _MISC_H
#define _MISC_H

char *_iodbcadm_getinifile (char *buf, int size, int bIsInst, int doCreate);
void _iodbcdm_getdsnfile(const char *filedsn, char *buf, size_t buf_sz);
const char *_iodbcdm_check_for_string (const char *szList,
    const char *szString, int bContains);
char *_iodbcdm_remove_quotes (const char *szString);
size_t _iodbcdm_strlcpy(char *dst, const char *src, size_t siz);
size_t _iodbcdm_strlcat(char *dst, const char *src, size_t siz);

extern WORD wSystemDSN;
extern WORD configMode;

#endif

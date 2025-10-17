/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 2, 2024.
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

#include <openssl/bio.h>

#if BIO_FLAGS_UPLINK==0
/* Shortcut UPLINK calls on most platforms... */
#define	UP_stdin	stdin
#define	UP_stdout	stdout
#define	UP_stderr	stderr
#define	UP_fprintf	fprintf
#define	UP_fgets	fgets
#define	UP_fread	fread
#define	UP_fwrite	fwrite
#undef	UP_fsetmod
#define	UP_feof		feof
#define	UP_fclose	fclose

#define	UP_fopen	fopen
#define	UP_fseek	fseek
#define	UP_ftell	ftell
#define	UP_fflush	fflush
#define	UP_ferror	ferror
#define	UP_fileno	fileno

#define	UP_open		open
#define	UP_read		read
#define	UP_write	write
#define	UP_lseek	lseek
#define	UP_close	close
#endif

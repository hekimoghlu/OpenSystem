/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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
#define	OLD_FILE	0
#define	NEW_FILE	1
#define	INDEX_FILE	2
#define	MAX_FILE	3

struct file_name {
#ifdef __APPLE__
	struct tm mtime;
	bool mtime_set;
#endif
	char *path;
	bool exists;
};

extern char	*source_file;

void		re_patch(void);
void		open_patch_file(const char *);
void		set_hunkmax(void);
bool		there_is_another_patch(void);
bool		another_hunk(void);
bool		pch_swap(void);
char		*pfetch(LINENUM);
size_t		pch_line_len(LINENUM);
LINENUM		pch_first(void);
LINENUM		pch_ptrn_lines(void);
LINENUM		pch_newfirst(void);
LINENUM		pch_repl_lines(void);
LINENUM		pch_end(void);
LINENUM		pch_context(void);
#ifdef __APPLE__
LINENUM		pch_leading_context(void);
LINENUM		pch_trailing_context(void);
#endif
LINENUM		pch_hunk_beg(void);
char		pch_char(LINENUM);
void		do_ed_script(void);

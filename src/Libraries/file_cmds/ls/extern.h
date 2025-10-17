/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 19, 2023.
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
#ifndef _LS_EXTERN_H_
#define _LS_EXTERN_H_

#include <stdbool.h>

int	 acccmp(const FTSENT *, const FTSENT *);
int	 revacccmp(const FTSENT *, const FTSENT *);
int	 birthcmp(const FTSENT *, const FTSENT *);
int	 revbirthcmp(const FTSENT *, const FTSENT *);
int	 modcmp(const FTSENT *, const FTSENT *);
int	 revmodcmp(const FTSENT *, const FTSENT *);
int	 namecmp(const FTSENT *, const FTSENT *);
int	 revnamecmp(const FTSENT *, const FTSENT *);
int	 statcmp(const FTSENT *, const FTSENT *);
int	 revstatcmp(const FTSENT *, const FTSENT *);
int	 sizecmp(const FTSENT *, const FTSENT *);
int	 revsizecmp(const FTSENT *, const FTSENT *);

void	 printcol(const DISPLAY *);
void	 printlong(const DISPLAY *);
int	 printname(const char *);
void	 printscol(const DISPLAY *);
void	 printstream(const DISPLAY *);
void	 usage(void);
int	 prn_normal(const char *);
size_t	 len_octal(const char *, int);
int	 prn_octal(const char *);
int	 prn_printable(const char *);
#ifdef COLORLS
void	 parsecolors(const char *cs);
void	 colorquit(int);

#ifdef __APPLE__
extern	bool	 unix2003_compat;
#else
#define	unix2003_compat	true
#endif

extern	char	*ansi_fgcol;
extern	char	*ansi_bgcol;
extern	char	*ansi_coloff;
extern	char	*attrs_off;
extern	char	*enter_bold;

extern int	 colorflag;
extern bool	 explicitansi;

#define	COLORFLAG_NEVER		0
#define	COLORFLAG_AUTO		1
#define	COLORFLAG_ALWAYS	2
#endif
extern int	termwidth;

#endif /* _LS_EXTERN_H_ */

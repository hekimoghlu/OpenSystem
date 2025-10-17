/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 19, 2025.
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
#ifndef _TERMCAP_H
#define _TERMCAP_H 1

#if __STDC__

extern int tgetent (char *buffer, const char *termtype);

extern int tgetnum (const char *name);
extern int tgetflag (const char *name);
extern char *tgetstr (const char *name, char **area);

extern char PC;
extern short ospeed;
extern void tputs (const char *string, int nlines, int (*outfun) (int));

extern char *tparam (const char *ctlstring, char *buffer, int size, ...);

extern char *UP;
extern char *BC;

extern char *tgoto (const char *cstring, int hpos, int vpos);

#else /* not __STDC__ */

extern int tgetent ();

extern int tgetnum ();
extern int tgetflag ();
extern char *tgetstr ();

extern char PC;
extern short ospeed;

extern void tputs ();

extern char *tparam ();

extern char *UP;
extern char *BC;

extern char *tgoto ();

#endif /* not __STDC__ */

#endif /* not _TERMCAP_H */

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 5, 2025.
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
 * parameter defaults
 */
#define	CLCNT		1
#define	INCHAR		'\t'
#define	INGAP		8
#define	OCHAR		'\t'
#define OGAP		8
#define	LINES		66
#define	NMWD		5
#define	NMCHAR		'\t'
#define	SCHAR		'\t'
#define	PGWD		72
#define SPGWD		512

/*
 * misc default values
 */
#define	HDFMT		"%s %s Page %d\n\n\n"
#define	HEADLEN		5
#define	TAILLEN		5
#define	TIMEFMTD	"%e %b %H:%M %Y"
#define	TIMEFMTM	"%b %e %H:%M %Y"
#define	FNAME		""
#define	LBUF		8192
#define	HDBUF		512

/*
 * structure for vertical columns. Used to balance cols on last page
 */
struct vcol {
	char *pt;		/* ptr to col */
	int cnt;		/* char count */
};

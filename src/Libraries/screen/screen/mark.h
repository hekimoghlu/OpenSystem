/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 8, 2025.
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
struct markdata
{
  struct win *md_window;/* pointer to window we are working on */
  struct acluser *md_user;	/* The user who brought us up */
  int	cx, cy;		/* cursor Position in WIN coords*/
  int	x1, y1;		/* first mark in WIN coords */
  int	second;		/* first mark dropped flag */
  int	left_mar, right_mar, nonl;
  int	rep_cnt;	/* number of repeats */
  int	append_mode;	/* shall we overwrite or append to copybuffer */
  int	write_buffer;	/* shall we do a KEY_WRITE_EXCHANGE right away? */
  int	hist_offset;	/* how many lines are on top of the screen */
  char	isstr[100];	/* string we are searching for */
  int	isstrl;
  char	isistr[200];	/* string of chars user has typed */
  int	isistrl;
  int	isdir;		/* current search direction */
  int	isstartpos;	/* position where isearch was started */
  int	isstartdir;	/* direction when isearch was started */
};


#define W2D(y) ((y) - markdata->hist_offset)
#define D2W(y) ((y) + markdata->hist_offset)


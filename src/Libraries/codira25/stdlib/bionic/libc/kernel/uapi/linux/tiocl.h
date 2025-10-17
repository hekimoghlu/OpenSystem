/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 30, 2025.
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
#ifndef _LINUX_TIOCL_H
#define _LINUX_TIOCL_H
#define TIOCL_SETSEL 2
#define TIOCL_SELCHAR 0
#define TIOCL_SELWORD 1
#define TIOCL_SELLINE 2
#define TIOCL_SELPOINTER 3
#define TIOCL_SELCLEAR 4
#define TIOCL_SELMOUSEREPORT 16
#define TIOCL_SELBUTTONMASK 15
struct tiocl_selection {
  unsigned short xs;
  unsigned short ys;
  unsigned short xe;
  unsigned short ye;
  unsigned short sel_mode;
};
#define TIOCL_PASTESEL 3
#define TIOCL_UNBLANKSCREEN 4
#define TIOCL_SELLOADLUT 5
#define TIOCL_GETSHIFTSTATE 6
#define TIOCL_GETMOUSEREPORTING 7
#define TIOCL_SETVESABLANK 10
#define TIOCL_SETKMSGREDIRECT 11
#define TIOCL_GETFGCONSOLE 12
#define TIOCL_SCROLLCONSOLE 13
#define TIOCL_BLANKSCREEN 14
#define TIOCL_BLANKEDSCREEN 15
#define TIOCL_GETKMSGREDIRECT 17
#endif

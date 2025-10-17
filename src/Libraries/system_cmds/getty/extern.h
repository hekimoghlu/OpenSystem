/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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
struct delayval;
struct termios;

extern	char **environ;
extern	char editedhost[];
extern	char hostname[];
extern	struct termios tmode, omode;
extern	struct gettyflags gettyflags[];
extern	struct gettynums gettynums[];
extern	struct gettystrs gettystrs[];

int	 adelay(int, struct delayval *);
const char *autobaud(void);
int	 delaybits(void);
void	 edithost(const char *);
void	 gendefaults(void);
void	 gettable(const char *);
void	 makeenv(char *[]);
const char *portselector(void);
void	 set_ttydefaults(int);
void	 setchars(void);
void	 setdefaults(void);
void	 set_flags(int);
int	 speed(int);
int	 getty_chat(char *, int, int);

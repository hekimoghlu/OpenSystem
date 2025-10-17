/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 17, 2024.
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
#pragma prototyped

#include	<ast.h>
#include	"ulimit.h"

/*
 * This is the list of resouce limits controlled by ulimit
 * This command requires getrlimit(), vlimit(), or ulimit()
 */

#ifndef _no_ulimit 

const char	e_unlimited[] = "unlimited";
const char*	e_units[] = { 0, "block", "byte", "Kibyte", "second" };

const int	shtab_units[] = { 1, 512, 1, 1024, 1 };

const Limit_t	shtab_limits[] =
{
"as",		"address space limit",	RLIMIT_AS,	0,		'M',	LIM_KBYTE,
"core",		"core file size",	RLIMIT_CORE,	0,		'c',	LIM_BLOCK,
"cpu",		"cpu time",		RLIMIT_CPU,	0,		't',	LIM_SECOND,
"data",		"data size",		RLIMIT_DATA,	0,		'd',	LIM_KBYTE,
"fsize",	"file size",		RLIMIT_FSIZE,	0,		'f',	LIM_BLOCK,
"locks",	"number of file locks",	RLIMIT_LOCKS,	0,		'x',	LIM_COUNT,
"memlock",	"locked address space",	RLIMIT_MEMLOCK,	0,		'l',	LIM_KBYTE,
"msgqueue",	"message queue size",	RLIMIT_MSGQUEUE,0,		'q',	LIM_KBYTE,
"nice",		"scheduling priority",	RLIMIT_NICE,	0,		'e',	LIM_COUNT,
"nofile",	"number of open files",	RLIMIT_NOFILE,	"OPEN_MAX",	'n',	LIM_COUNT,
"nproc",	"number of processes",	RLIMIT_NPROC,	"CHILD_MAX",	'u',	LIM_COUNT,
"pipe",		"pipe buffer size",	RLIMIT_PIPE,	"PIPE_BUF",	'p',	LIM_BYTE,
"rss",		"max memory size",	RLIMIT_RSS,	0,		'm',	LIM_KBYTE,
"rtprio",	"max real time priority",RLIMIT_RTPRIO,	0,		'r',	LIM_COUNT,
"sbsize",	"socket buffer size",	RLIMIT_SBSIZE,	"PIPE_BUF",	'b',	LIM_BYTE,
"sigpend",	"signal queue size",	RLIMIT_SIGPENDING,"SIGQUEUE_MAX",'i',	LIM_COUNT,
"stack",	"stack size",		RLIMIT_STACK,	0,		's',	LIM_KBYTE,
"swap",		"swap size",		RLIMIT_SWAP,	0,		'w',	LIM_KBYTE,
"threads",	"number of threads",	RLIMIT_PTHREAD,	"THREADS_MAX",	'T',	LIM_COUNT,
"vmem",		"process size",		RLIMIT_VMEM,	0,		'v',	LIM_KBYTE,
{ 0 }
};

#endif

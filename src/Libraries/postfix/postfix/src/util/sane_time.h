/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 2, 2022.
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

#ifndef _SANE_TIME_H_
#define _SANE_TIME_H_

/*++
/* NAME
/*	sane_time 3h
/* SUMMARY
/*	time(2) with backward time jump protection
/* SYNOPSIS
/*	#include <sane_time.h>
/* DESCRIPTION
/* .nf

 /*
  * System library.
  */
#include <time.h>

 /*
  * External interface.
  */
extern time_t sane_time(void);

/* LICENSE
/* .ad
/* .fi
/*	The Secure Mailer license must be distributed with this software.
/* AUTHOR(S)
/*	Patrik Rak
/*	Modra 6
/*	155 00, Prague, Czech Republic
/*--*/

#endif

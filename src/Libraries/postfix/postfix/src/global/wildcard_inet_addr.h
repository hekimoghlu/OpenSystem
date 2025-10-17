/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 29, 2024.
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

#ifndef _WILDCARD_INET_ADDR_H_INCLUDED_
#define _WILDCARD_INET_ADDR_H_INCLUDED_

/*++
/* NAME
/*	wildcard_inet_addr 3h
/* SUMMARY
/*	grab the list of wildcard IP addresses.
/* SYNOPSIS
/*	#include <wildcard_inet_addr.h>
/* DESCRIPTION
/* .nf
/*--*/

 /*
  * Utility library.
  */
#include <inet_addr_list.h>

 /*
  * External interface.
  */
extern struct INET_ADDR_LIST *wildcard_inet_addr_list(void);

/* LICENSE
/* .ad
/* .fi
/*	foo
/* AUTHOR(S)
/*	Jun-ichiro itojun Hagino
/*--*/

#endif
